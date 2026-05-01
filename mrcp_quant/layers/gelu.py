import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from mrcp_quant.observers import MinMaxObserver
from mrcp_quant.quant_utils import new_ln, TaylorExponent
from mrcp_quant.layers.hadamard import qHadamardProd

class IntGeluTS(nn.Module):
    """
    Quantized GELU using a combination of Lookup Table (LUT) approximation and Taylor Series expansion.
    This class uses integer-only arithmetic to perform the GELU function with quantized inputs.
    """

    def __init__(self,
                 quant=False,
                 LUT_SIZE=16,
                 nof_bits=16,
                 hidden_act='gelu',
                 eps=1e-5,
                 gelu_scale=1.702,
                 is_opt_scale=False,
                 is_calibrate=False,
                 ts_ln=new_ln,
                 int_exp_scaled=TaylorExponent):
        super(IntGeluTS, self).__init__()

        self.LUT_SIZE = LUT_SIZE
        self.nof_bits = nof_bits
        self.iterations = 2
        self.is_opt_scale = is_opt_scale
        self.is_weights_quantized = True
        self.gelu_scale = gelu_scale
        self.ts_ln = ts_ln
        self.int_exp_scaled = int_exp_scaled
        self.eps = eps

        self.input_bits = self.nof_bits - 4
        self.output_bits = self.nof_bits + 1

        # self.ln2 = (torch.tensor(2).log() * 2 ** (self.output_bits-1)).round().to(torch.int32)

        self.quant = quant
        self.is_calibrate = is_calibrate

        self.had_mul = qHadamardProd(nof_bits1=nof_bits,
                                     nof_bits2=nof_bits,
                                     quant=quant)

        self.in_obs = MinMaxObserver(qscheme=torch.per_tensor_affine,
                                     nof_bits=nof_bits,
                                     is_calibrate=True,
                                     name="geLU")

        self.k_values = torch.tensor([
            2.834596, 2.338217, 1.978175, 1.754823, 1.642511,
            1.642511, 1.754823, 1.978175, 2.338217, 2.834596
        ]).cuda()

        self.exp_lut = []
        self.ln_lut =  []
        self.fp_gelu = ACT2FN[hidden_act]   # pass in config or string

        # self.exp_lut = build_exp_lut(nof_bits=self.output_bits, LUT_SIZE=self.LUT_SIZE)
        # self.ln_lut = build_ln_lut(nof_bits=self.output_bits, LUT_SIZE=self.LUT_SIZE, eps=self.eps)

        # if torch.cuda.is_available():
        #     self.exp_lut = self.exp_lut.cuda()
        #     self.ln_lut = self.ln_lut.cuda()
        #     self.k_values = self.k_values.cuda()

    def int_sigmoid(self, x, nof_bits=16, iterations=2, gelu_scale=1.702, LUT_SIZE=-1):
        input_bits = nof_bits - 4
        output_bits = nof_bits

        x_sig = x * gelu_scale
        x_max = torch.clamp(x_sig, min=0)
        x_int = x_sig - x_max

        spacial_scale = 1 << (input_bits - 1)
        x_scale = (x_int * spacial_scale).floor().to(dtype=torch.int32)
        x_max_scale = -(x_max * spacial_scale).floor().to(dtype=torch.int32)

        exp_int = TaylorExponent(x_scale,
                              spacial_scale,
                              input_bits=input_bits,
                              output_bits=output_bits,
                              LUT_SIZE=LUT_SIZE,
                              exp_lut=[],
                              iterations=0)
                              # iterations=iterations)

        exp_zero = TaylorExponent(x_max_scale,
                              spacial_scale,
                              input_bits=input_bits,
                              output_bits=output_bits,
                              LUT_SIZE=LUT_SIZE,
                              exp_lut=[],
                              iterations=iterations)

        exp_int_sum = exp_int + exp_zero

        return exp_int, exp_int_sum

    def get_k_value(self, x):
        x = torch.clamp(x, -5, 5)
        x_shifted = x + 5
        indices = torch.clamp(x_shifted.floor().long(), 0, 9)
        return self.k_values[indices]

    def forward_pass(self, x):
        alpha = self.get_k_value(x)
        # alpha = 1.702  # self.get_k_value(x)

        exp_int, exp_int_sum = self.int_sigmoid(x, self.nof_bits, self.iterations, alpha, self.LUT_SIZE)
        ln_sum = new_ln(exp_int_sum, self.output_bits-1)

        # ln_sum = self.ts_ln(exp_int_sum,
        #                     iterations=self.iterations+1,
        #                     nof_bits=self.output_bits,
        #                     ln_lut=self.ln_lut,
        #                     LN2=None)

        spacial_scale_out = 1 << (self.output_bits - 1)

        ln_mul = self.int_exp_scaled(-ln_sum,
                                     spacial_scale_out,
                                     input_bits=self.output_bits,
                                    output_bits=self.output_bits,
                                    LUT_SIZE=self.LUT_SIZE,
                                    exp_lut=[],
                                    iterations=self.iterations)

        q_sigmoid = ln_mul * exp_int
        deq_sigmoid = q_sigmoid / (1 << ((self.output_bits - 1) + (self.output_bits - 1)))

        return self.had_mul(x, deq_sigmoid)

    def float_forward_pass(self, x):
        normal = torch.distributions.Normal(0.0, 1.0)
        sig = normal.cdf(x) # collect data for other observers
        _ = self.had_mul(x, sig) # collect data for other observers
        # return self.had_mul(x, sig)
        return self.fp_gelu(x)



    def quantize_weights_and_bias(self):
        if not self.is_weights_quantized:
            self.is_weights_quantized = True

    def forward(self, x):
        if self.is_calibrate:
            self.in_obs(x)
            return self.float_forward_pass(x)

        if self.is_opt_scale:
            self.opt_input = x
            return self.float_forward_pass(x)

        return self.forward_pass(x) if self.quant else self.float_forward_pass(x)

    def set_calibration_flag(self):
        self.is_calibrate = True

    def unset_calibration_flag(self):
        self.is_calibrate = False

    def set_quant(self):
        self.quant = True

    def unset_quant(self):
        self.quant = False

    def set_scale_opt(self):
        self.is_opt_scale = True

    def unset_scale_opt(self):
        del self.opt_input
        self.is_opt_scale = False

"""## Layer norm"""
