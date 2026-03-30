"""Quantized LayerNorm implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptq_tr.quantization.observers.minmax import MinMaxObserver
from ptq_tr.quantization.utils import TaylorExponent, TylorNLog, build_exp_lut, build_ln_lut, new_ln


class QLayerNorm(nn.LayerNorm):
    """
    A subclass of PyTorch's LayerNorm with quantization support.

    Args:
        normalized_shape: Input shape from an expected input of size (N, *), where * means any number of additional dimensions.
        eps: A value added to the denominator for numerical stability.
        elementwise_affine: When set to True, this module has learnable per-element affine parameters.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        in1_bits=16,
        in2_bits=16,
        LUT_SIZE=16,
        quant=False,
        elementwise_affine=True,
        is_opt_scale=False,
        is_calibrate=False,
        ts_ln=TylorNLog,
        int_exp_scaled=TaylorExponent,
        iterations=3,
    ):
        super(QLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.ts_ln = ts_ln
        self.int_exp_scaled = int_exp_scaled

        self.in1_bits = in1_bits
        self.in2_bits = in2_bits - 1
        self.is_opt_scale = is_opt_scale
        self.is_calibrate = is_calibrate
        self.LUT_SIZE = LUT_SIZE
        self.quant = quant
        self.is_weights_quantized = False
        self.iterations = iterations
        self.input_bits1 = self.in1_bits
        self.output_bits1 = self.in1_bits

        self.in_obs_normalize = MinMaxObserver(
            qscheme=torch.per_tensor_affine,
            nof_bits=self.input_bits1,
            is_calibrate=True,
            name="lnorm",
        )

        self.in_obs = MinMaxObserver(
            qscheme=torch.per_tensor_affine,
            nof_bits=self.input_bits1,
            is_calibrate=True,
            name="lnorm",
        )
        self.w_obs = MinMaxObserver(
            qscheme=torch.per_tensor_affine,
            nof_bits=self.in2_bits,
            is_calibrate=True,
            name="lnorm",
        )

        if self.bias is not None:
            self.b_obs = MinMaxObserver(
                qscheme=torch.per_tensor_symmetric,
                nof_bits=self.in1_bits + self.in2_bits,
                is_calibrate=True,
                name="lnorm",
            )
        else:
            self.b_obs = None

        self.output_bits1 = self.in1_bits
        self.exp_lut = build_exp_lut(nof_bits=self.output_bits1, LUT_SIZE=self.LUT_SIZE)
        self.ln_lut = build_ln_lut(nof_bits=self.output_bits1, LUT_SIZE=self.LUT_SIZE, eps=self.eps)
        self.ln2 = (torch.tensor(2).log() * 2 ** (self.output_bits1 - 1)).round().to(torch.int32)
        self.stats = dict()

        self.stats["scale"] = []
        self.stats["ref"] = []
        self.stats["var"] = torch.tensor([]).cuda()
        self.sample_counter = 32

        if torch.cuda.is_available():
            self.exp_lut = self.exp_lut.cuda()
            self.ln_lut = self.ln_lut.cuda()

    def scaled_ln16(self, in_x, nof_bits=8):
        scale_factor = 1 << (self.input_bits1 - 1)
        scaled_input = (in_x * scale_factor).floor().to(dtype=torch.int32)

        ln_sum = new_ln(scaled_input, self.output_bits1 - 1)

        out = TaylorExponent(
            -ln_sum >> 1,
            scale_factor,
            iterations=2,
            input_bits=self.output_bits1,
            output_bits=self.output_bits1,
        )

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
            self.w_obs(self.weight)
            self.scale_weight, self.zero_point_weight = self.w_obs.calculate_qparams()
            self.weight_integer = self.w_obs.quantizer(self.weight)
            if self.bias is not None:
                self.b_obs.scale = self.in_obs.scale * self.w_obs.scale
                self.b_obs.zero_point = 0
                self.bias_integer = self.b_obs.quantizer(self.bias)
            else:
                self.b_obs.scale = self.in_obs.scale * self.w_obs.scale
                self.b_obs.zero_point = 0
                self.bias_integer = None

            self.is_weights_quantized = True

    def betta_gamma_forward_pass(self, x):
        """
        Forward pass with quantization:
        - Quantizes weights and bias if not already done.
        - Quantizes the input.
        - Performs linear transformation with quantized weights and bias.
        - Dequantizes the output before returning.
        """
        self.quantize_weights_and_bias()

        _, self.zero_point_input = self.in_obs.calculate_qparams()
        x_q = self.in_obs.quantizer(x)
        x_q = x_q - self.zero_point_input.int()

        if self.bias is not None:
            output_q = x_q * (self.weight_integer - self.zero_point_weight.int()) + self.bias_integer
        else:
            output_q = x_q * (self.weight_integer - self.zero_point_weight.int())

        dq_output = self.b_obs.dequantizer(output_q)
        return dq_output

    def normalize(self, x):
        self.scale_input_normed, self.zero_point_input_normed = self.in_obs_normalize.calculate_qparams()

        x_q = self.in_obs_normalize.quantizer(x)
        x_q = x_q - self.zero_point_input_normed

        xq_mean = x_q.float().mean(dim=-1, keepdim=True).long()
        xq_var = x_q.float().var(dim=-1, unbiased=False, keepdim=True)
        x_var = ((self.scale_input_normed) ** 2) * (xq_var.float())

        ln_mul = self.scaled_ln16(x_var, nof_bits=self.in1_bits)

        q_mean_val = x_q - xq_mean
        x_dq = ln_mul * q_mean_val * ((self.scale_input_normed) / 2 ** (self.in1_bits - 1))

        return x_dq

    def float_betta_gamma_forward_pass(self, x):
        return x * self.weight + self.bias

    def forward_pass(self, x):
        mean_val = self.normalize(x)
        dq_output = self.betta_gamma_forward_pass(mean_val)
        return dq_output

    def float_forward_pass(self, x):
        if self.sample_counter < 32:
            self.stats["var"] = torch.cat((self.stats["var"], x.var(dim=-1, unbiased=False, keepdim=True)), dim=-1)
            self.sample_counter += 1

        return F.layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

    def forward(self, x):
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
