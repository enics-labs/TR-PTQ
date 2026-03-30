"""Patch merging implementation."""

import torch

from ptq_tr.quantization.modules.q_layernorm import QLayerNorm
from ptq_tr.quantization.modules.quant_linear import QuantizedLinear
from ptq_tr.quantization.qparams import QauntParams


class PatchMerging(QauntParams):
    r"""Patch Merging Layer."""

    def __init__(
        self,
        input_resolution,
        dim,
        quant=None,
        quant_config=None,
        norm_layer=QLayerNorm,
    ):
        super().__init__(quant_config=quant_config)
        quant = self.quant if quant is None else quant
        self.quant = quant
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = QuantizedLinear(
            4 * dim,
            2 * dim,
            bias=False,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )

        self.norm = norm_layer(
            4 * dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            quant=quant,
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops
