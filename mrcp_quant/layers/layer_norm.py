import torch
import torch.nn as nn
import torch.nn.functional as F

from mrcp_quant.observers import MinMaxObserver
from mrcp_quant.quant_utils import build_exp_lut, build_ln_lut, new_ln, TaylorExponent

class QLayerNorm(nn.LayerNorm):
    """
    A subclass of PyTorch's LayerNorm with quantization support.

    Args:
        normalized_shape: Input shape from an expected input of size (N, *), where * means any number of additional dimensions.
        eps: A value added to the denominator for numerical stability.
        elementwise_affine: When set to True, this module has learnable per-element affine parameters.
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-5,
                 in1_bits=16,
                 in2_bits=16,
                 LUT_SIZE=16,
                 quant=False,
                 elementwise_affine=True,
                 is_opt_scale=False,
                 is_calibrate=False,
                 ts_ln=new_ln,
                 int_exp_scaled=TaylorExponent,
                 split_table=-1,
                 iterations=3):

        super(QLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.ts_ln = ts_ln
        self.int_exp_scaled = int_exp_scaled
        self.split_table = split_table

        self.in1_bits = in1_bits
        self.in2_bits = in2_bits-1
        self.is_opt_scale = is_opt_scale
        self.is_calibrate = is_calibrate
        self.LUT_SIZE = LUT_SIZE
        self.quant = quant  # Control whether to perform quantized operations
        self.is_weights_quantized = False  # Track whether weights have been quantized
        self.iterations = iterations
        self.input_bits1 = self.in1_bits
        self.output_bits1 = self.in1_bits


        # self.in_obs_normalize = PerChannelMinMaxObserver(torch.per_channel_affine,
        self.in_obs_normalize = MinMaxObserver(qscheme=torch.per_tensor_affine,
                                     nof_bits=self.input_bits1,
                                    #  is_calibrate=True,
                                     name="lnorm")

        # self.in_obs_normalize = PercentileMinMaxObserver(qscheme=torch.per_tensor_affine,
        #                                                  nof_bits=self.input_bits1,
        #                                                  is_calibrate=True,
        #                                                  name="lnorm")

        # self.in_obs_normalize = HistogramObserver(qscheme=torch.per_tensor_affine,
        #                                 nof_bits=self.input_bits1,
        #                                 is_calibrate=True,
        #                                 bins=512,
        #                                 name="lnorm")

        # self.in_obs_normalize = PerChannelMinMaxObserver(
        #     ch_axis=-1,
        #     qscheme=torch.per_channel_affine,
        #     nof_bits=self.input_bits1,
        #     name="lnorm"
        # )

        # Initialize observers for inputs, weights, and self.bias
        self.in_obs = MinMaxObserver(qscheme=torch.per_tensor_affine,
                                     nof_bits=self.input_bits1,
                                     is_calibrate=True,
                                     name="lnorm")
        self.w_obs = MinMaxObserver(qscheme=torch.per_tensor_affine,
                                    nof_bits=self.in2_bits,
                                    is_calibrate=True,
                                    name="lnorm")

        if self.bias is not None:
            self.b_obs = MinMaxObserver(qscheme=torch.per_tensor_symmetric,
                                      nof_bits=self.in1_bits+self.in2_bits,
                                      is_calibrate=True,
                                      name="lnorm")
        else:
            self.b_obs = None

        self.output_bits1 = self.in1_bits
        self.exp_lut = build_exp_lut(nof_bits=self.output_bits1, LUT_SIZE=self.LUT_SIZE)
        self.ln_lut = build_ln_lut(nof_bits=self.output_bits1, LUT_SIZE=self.LUT_SIZE, eps=self.eps)
        self.ln2 = (torch.tensor(2).log() * 2 ** (self.output_bits1-1)).round().to(torch.int32)
        self.stats = dict()

        self.s1 = 1
        self.s2 = 1
        # self.sample_counter = 0
        self.sample_counter = 32 # do not sample

        if torch.cuda.is_available():
            self.exp_lut = self.exp_lut.cuda()
            self.ln_lut = self.ln_lut.cuda()

    # Define ts_ln function
    def scaled_ln16(self, in_x, nof_bits=8):
        scale_factor = 1 << (self.input_bits1 - 1)
        scaled_input = (in_x * scale_factor).floor().to(dtype=torch.int32)

        # log_factor = torch.where(
        #     scaled_input >> (self.output_bits1 - 1) == 0,
        #     2,
        #     0
        # )


        # ln_sum = self.ts_ln(scaled_input << (log_factor << 1),
        #                     iterations=3,
        #                     nof_bits=self.output_bits1,
        #                     ln_lut=self.ln_lut,
        #                     LN2=self.ln2) >> 1


        # out = self.int_exp_scaled(-ln_sum,
        #                           scale_factor,
        #                           input_bits=self.output_bits1,
        #                           output_bits=self.output_bits1,
        #                           LUT_SIZE=self.LUT_SIZE,
        #                           exp_lut=self.exp_lut,
        #                           iterations=2) << log_factor

        ln_sum = new_ln(scaled_input, self.output_bits1-1)

        out = TaylorExponent(-ln_sum>>1,
                             scale_factor,
                             iterations=2,
                             input_bits=self.output_bits1,
                             output_bits=self.output_bits1,
                             split_table=self.split_table)

        return out

    def float_foward_pass_normalize(self, x):
        x_mean = x.float().mean(dim=-1, keepdim=True)
        x_var = x.float().var(dim=-1, unbiased=False, keepdim=True)
        x_div = x_var.sqrt()
        normed_x = (x - x_mean) / x_div
        return normed_x

    def quantize_weights_and_bias(self):
        """
        Quantizes weights and bias if they have not been quantized yet.
        Sets scale and zero-point for weights and bias, and stores quantized values.
        """
        if not self.is_weights_quantized:
            # Quantize the weights
            self.w_obs(self.weight)
            self.scale_weight, self.zero_point_weight = self.w_obs.calculate_qparams()
            self.weight_integer = self.w_obs.quantizer(self.weight)
            # print(f"[DEBUG] : self.zero_point_weight: {self.zero_point_weight}, min val: {self.w_obs.min_val}, max val: {self.w_obs.max_val}, scale: {self.scale_weight}")
            # Quantize the bias if it exists
            if self.bias is not None:
                # Calculate bias quantization scale as the product of input and weight scales
                self.b_obs.scale = self.in_obs.scale * self.w_obs.scale
                self.b_obs.zero_point = 0  # bias zero-point is typically set to 0
                self.bias_integer = self.b_obs.quantizer(self.bias)
            else:
                # Set scale and zero-point for cases without a bias
                self.b_obs.scale = self.in_obs.scale * self.w_obs.scale
                self.b_obs.zero_point = 0
                self.bias_integer = None

            # Mark weights and bias as quantized
            self.is_weights_quantized = True

    def betta_gamma_forward_pass(self, x):
        """
        Forward pass with quantization:
        - Quantizes weights and bias if not already done.
        - Quantizes the input.
        - Performs linear transformation with quantized weights and bias.
        - Dequantizes the output before returning.
        """

        # Ensure weights and bias are quantized
        self.quantize_weights_and_bias()

        # Quantize the input
        _ , self.zero_point_input = self.in_obs.calculate_qparams()
        # _ , self.zero_point_weight = self.w_obs.calculate_qparams()
        x_q = self.in_obs.quantizer(x)  # Quantize the input
        x_q = x_q - self.zero_point_input.int()  # Adjust input by zero point

        # Integer linear transformation using quantized weights and bias
        if self.bias is not None:
            output_q = x_q * (self.weight_integer - self.zero_point_weight.int()) + self.bias_integer
        else:
            output_q = x_q * (self.weight_integer - self.zero_point_weight.int())

        # print(self.w_obs.dequantizer(self.weight_integer + self.zero_point_weight.int()))
        # print("-----------------")
        # print(self.weight)
        # print("=================")

        # Dequantize the output
        dq_output = self.b_obs.dequantizer(output_q)
        return dq_output

    def normalize(self, x):
        self.x_test = x
        self.scale_input_normed , self.zero_point_input_normed = self.in_obs_normalize.calculate_qparams()

        x_q = self.in_obs_normalize.quantizer(x)
        x_q = (x_q - self.zero_point_input_normed).int()

        # xq_mean = x_q.float().mean(dim=-1, keepdim=True).long()
        # xq_var = x_q.float().var(dim=-1, unbiased=False, keepdim=True)
        xq_sum = x_q.sum(dim=-1, keepdim=True)
        xq_scaled_mean = (xq_sum*self.s1) / x_q.shape[-1]   # float division
        xq_mean = xq_scaled_mean.trunc().int()

        q_e_2_x = xq_mean * xq_mean
        q_e_x_2 = (((x_q**2).sum(dim=-1, keepdim=True) * self.s2) / x_q.shape[-1]).trunc().int()
        q_var_x = q_e_x_2 - q_e_2_x

        # dq_xq_mean =  xq_mean * m.in_obs_normalize.scale / 2**self.s1


        x_var = ((self.scale_input_normed) ** 2) * (q_var_x.float()) / self.s2
        # x_var = ((self.scale_input_normed) ** 2) * (xq_var.float())

        ln_mul = self.scaled_ln16(x_var, nof_bits=self.in1_bits)

        q_mean_val = (x_q - xq_mean)
        x_dq = ln_mul * q_mean_val * ((self.scale_input_normed) / 2**(self.in1_bits-1)) / self.s1

        # x_mean = x.float().mean(dim=-1, keepdim=True)
        # x_var = x.float().var(dim=-1, unbiased=False, keepdim=True)
        # x_div = x_var.sqrt()

        return x_dq

    def float_betta_gamma_forward_pass(self, x):
        return x * self.weight + self.bias

    def forward_pass(self, x):
        mean_val = self.normalize(x)
        # mean_val = self.float_foward_pass_normalize(x)
        dq_output = self.betta_gamma_forward_pass(mean_val)
        # dq_output = self.float_betta_gamma_forward_pass(mean_val)
        # self.stats['var'].append(dq_output)
        # self.stats['ref'].append(self.float_betta_gamma_forward_pass(mean_val))

        return dq_output

    def float_forward_pass(self, x):
        # if self.sample_counter < 32:
        #     self.stats['var'] = torch.cat((self.stats['var'], x.var(dim=-1, unbiased=False, keepdim=True)), dim=-1)
        #     self.sample_counter += 1

        return F.layer_norm(x,
                            self.normalized_shape,
                            weight=self.weight,
                            bias=self.bias,
                            eps=self.eps
        )

    def forward(self, x):

        # Calibration mode: update the input observer and return the input tensor as-is
        if self.is_calibrate:
            self.in_obs_normalize(x)
            normed_x = self.float_foward_pass_normalize(x)
            _ = self.in_obs(normed_x)
            return self.float_forward_pass(x)

        if self.is_opt_scale:
            self.opt_input = x
            return self.float_forward_pass(x)

        if self.quant:
            dq_output = self.forward_pass(x)
            return dq_output
        else:
            return self.float_forward_pass(x)

    def set_calibration_flag(self):
        """Enable calibration mode to update observers."""
        self.is_calibrate = True

    def unset_calibration_flag(self):
        """Disable calibration mode."""
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
