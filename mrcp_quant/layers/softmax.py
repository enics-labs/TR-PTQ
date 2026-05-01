import torch
import torch.nn as nn

from mrcp_quant.observers import MinMaxObserver
from mrcp_quant.quant_utils import new_ln, TaylorExponent

class IntSoftmaxTS(nn.Module):
    """
    Quantized Softmax with a combination of Lookup Table (LUT) approximation and Taylor Series expansion.
    This class uses integer-only arithmetic to perform the softmax function with quantized inputs.
    """

    def __init__(self,
                 quant=False,
                 is_calibrate=False,
                 nof_bits=16,
                 LUT_SIZE=16,
                 is_opt_scale=False,
                 eps=1e-5,
                 dim=-1,
                 iterations=2,
                 split_table=0,
                 ts_ln=new_ln,
                 int_exp_scaled=TaylorExponent):
        super(IntSoftmaxTS, self).__init__()

        self.LUT_SIZE = LUT_SIZE
        self.nof_bits = nof_bits
        self.iterations = iterations
        self.split_table = split_table
        self.dim = dim
        self.quant = quant
        self.is_calibrate = is_calibrate
        self.is_opt_scale = is_opt_scale
        self.eps = eps

        scale = 1 << (self.nof_bits - 1)
        self.div3 = int(round(scale / 3))

        self.ts_ln = ts_ln
        self.int_exp_scaled = int_exp_scaled

        self.input_bits = self.nof_bits - 4
        self.output_bits = self.nof_bits + 1
        self.stats = dict()
        self.stats[f'softmax'] = []

        self.ln2 = (torch.tensor(2).log() * 2 ** (self.output_bits-1)).round().to(torch.int32)

        self.in_obs = MinMaxObserver(qscheme=torch.per_tensor_symmetric,
                                     nof_bits=self.nof_bits,
                                     is_calibrate=is_calibrate,
                                     name="Softmax")

        self.exp_lut = []
        self.ln_lut =  []

        # self.exp_lut = build_exp_lut(nof_bits=self.output_bits, LUT_SIZE=self.LUT_SIZE)
        # self.ln_lut = build_ln_lut(nof_bits=self.output_bits, LUT_SIZE=self.LUT_SIZE, eps=self.eps)

        # if torch.cuda.is_available():
        #     self.exp_lut = self.exp_lut.cuda()
        #     self.ln_lut = self.ln_lut.cuda()



    def int_softmax(self, x):

        dtype = torch.int32
        udtype = torch.int32

        self.scale_input, self.zero_point_input = self.in_obs.calculate_qparams()
        # try:
        #     print(f"self.scale_test: {self.scale_test}")
        # except Exception as err:
        #     print("a test")
        #     self.scale_test = self.scale_input
        #     print(f"self.in_obs.min_val: {self.in_obs.min_val}")
        #     print(f"self.in_obs.max_val: {self.in_obs.max_val}")

        # print(f"x.min(): {x.min()}")
        # print(f"x.max(): {x.max()}")
        # print(f"x.round().min(): {x.round().min()}")
        # print(f"x.round().max(): {x.round().max()}")
        _x_q = self.in_obs.quantizer(x)  # Quantize the input
        x_q = _x_q  # Adjust input by zero point

        # reference
        # x_int = x - x.max(dim=self.dim, keepdim=True).values

        # rounding float
        # x_round = x.round()
        # x_int = x_round - x_round.max(dim=self.dim, keepdim=True).values

        # quantized 4 bits
        x_int = self.in_obs.dequantizer(x_q - x_q.max(dim=self.dim, keepdim=True).values).round()

        spacial_scale = 1 << (self.input_bits - 1)
        x_scale = (x_int * spacial_scale).floor().to(dtype)

        try:
            exp_int = self.int_exp_scaled(x_scale,
                                          spacial_scale,
                                          input_bits=self.input_bits,
                                          output_bits=self.output_bits,
                                          LUT_SIZE=self.LUT_SIZE,
                                          exp_lut=self.exp_lut,
                                          split_table=self.split_table,
                                          iterations=1)  # TODO test zero iterations
                                          # iterations=self.iterations)
        except RuntimeError as err:
            print("[EXP 1] : TEST")
            print(x_int)
            raise err
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        ln_sum = new_ln(exp_int_sum, self.output_bits-1)
        # self.ts_ln(exp_int_sum,
        #                     iterations=self.iterations+1,
        #                     nof_bits=self.output_bits,
        #                     ln_lut=self.ln_lut,
        #                     LN2=self.ln2)

        spacial_scale = 1 << (self.output_bits - 1)

        try:
            ln_mul = self.int_exp_scaled(-ln_sum.int(),
                                        spacial_scale,
                                        input_bits=self.output_bits,
                                        output_bits=self.output_bits,
                                        LUT_SIZE=self.LUT_SIZE,
                                        exp_lut=self.exp_lut,
                                        iterations=self.iterations)
        except RuntimeError as err:
            print("[EXP 2] : TEST")
            print(x_int)
            raise err

        sf_values = ln_mul * exp_int
        return sf_values

    def foward_pass(self, x):
        sf_values = self.int_softmax(x)
        deq_softmax = sf_values / (1 << ((self.output_bits - 1) + (self.output_bits - 1)))
        return deq_softmax

    def float_forward_pass(self, x):
        return x.softmax(dim=self.dim)


    def forward(self, x):
        if self.is_calibrate:
            self.in_obs(x);  return x.softmax(dim=self.dim)

        if self.is_opt_scale:
            self.opt_input = x;  return x.softmax(dim=self.dim)

        # self.stats[f'softmax'].append(x)

        if self.quant:
            return self.foward_pass(x)
        else:
            return self.float_forward_pass(x)

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

"""## Hadamard mul (Mul by Element)"""
