"""Integer softmax implementation."""

import torch
import torch.nn as nn

from ptq_tr.quantization.observers.minmax import MinMaxObserver
from ptq_tr.quantization.utils import TaylorExponent, TylorNLog, new_ln


sf_layer_idx = 0


class IntSoftmaxTS(nn.Module):
    """
    Quantized Softmax with a combination of Lookup Table (LUT) approximation and Taylor Series expansion.
    This class uses integer-only arithmetic to perform the softmax function with quantized inputs.
    """

    def __init__(
        self,
        quant=False,
        is_calibrate=False,
        nof_bits=16,
        LUT_SIZE=16,
        is_opt_scale=False,
        eps=1e-5,
        dim=-1,
        iterations=1,
        int_bits=4,
        ts_ln=TylorNLog,
        int_exp_scaled=TaylorExponent,
    ):
        super(IntSoftmaxTS, self).__init__()

        self.LUT_SIZE = LUT_SIZE
        self.nof_bits = nof_bits
        self.iterations = iterations
        self.dim = dim
        self.int_bits = int_bits
        self.quant = quant
        self.is_calibrate = is_calibrate
        self.is_opt_scale = is_opt_scale
        self.eps = eps

        if not (1 <= self.int_bits <= self.nof_bits):
            raise ValueError(f"int_bits must be in [1, {self.nof_bits}], got {self.int_bits}")

        scale = 1 << (self.nof_bits - 1)
        self.div3 = int(round(scale / 3))

        self.ts_ln = ts_ln
        self.int_exp_scaled = int_exp_scaled

        # Split total precision into integer and fractional fixed-point bits.
        self.frac_bits = self.nof_bits - self.int_bits
        self.output_bits = self.nof_bits + 1
        self.stats = dict()
        self.stats["softmax"] = []

        self.ln2 = (torch.tensor(2).log() * 2 ** (self.output_bits - 1)).round().to(torch.int32)

        self.in_obs = MinMaxObserver(
            qscheme=torch.per_tensor_symmetric,
            nof_bits=self.nof_bits,
            is_calibrate=is_calibrate,
            name="Softmax",
        )

        self.exp_lut = []
        self.ln_lut = []

    def int_softmax(self, x):
        dtype = torch.int32
        udtype = torch.int32

        self.scale_input, self.zero_point_input = self.in_obs.calculate_qparams()
        _x_q = self.in_obs.quantizer(x)
        x_q = _x_q

        x_int = self.in_obs.dequantizer(x_q - x_q.max(dim=self.dim, keepdim=True).values).round()

        spacial_scale = 1 << (self.frac_bits - 1)
        x_scale = (x_int * spacial_scale).floor().to(dtype)

        try:
            exp_int = self.int_exp_scaled(
                x_scale,
                spacial_scale,
                input_bits=self.frac_bits,
                output_bits=self.output_bits,
                LUT_SIZE=self.LUT_SIZE,
                exp_lut=self.exp_lut,
                iterations=0,
            )
        except RuntimeError as err:
            raise err

        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        ln_sum = new_ln(exp_int_sum, self.output_bits - 1)

        spacial_scale = 1 << (self.output_bits - 1)

        try:
            ln_mul = self.int_exp_scaled(
                -ln_sum.int(),
                spacial_scale,
                input_bits=self.output_bits,
                output_bits=self.output_bits,
                LUT_SIZE=self.LUT_SIZE,
                exp_lut=self.exp_lut,
                iterations=1,
            )
        except RuntimeError as err:
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
            self.in_obs(x)
            return x.softmax(dim=self.dim)

        if self.is_opt_scale:
            self.opt_input = x
            return x.softmax(dim=self.dim)

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
