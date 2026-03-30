"""Attention block implementation."""

import torch.nn as nn

from ptq_tr.quantization.modules.int_softmax import IntSoftmaxTS
from ptq_tr.quantization.modules.quant_linear import QuantizedLinear
from ptq_tr.quantization.modules.quant_matmul import QuantizedMatmul
from ptq_tr.quantization.qparams import QauntParams


class Attention(QauntParams):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        quant=None,
        quant_config=None,
    ):
        super().__init__(quant_config=quant_config)
        quant = self.quant if quant is None else quant
        self.quant = quant
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = QuantizedLinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.mat_mul_qk = QuantizedMatmul(
            in1_bits=self.nof_bits_matmul1,
            in2_bits=self.nof_bits_matmul2,
            quant=quant,
        )
        self.sf = IntSoftmaxTS(
            nof_bits=self.nof_bits_softmax,
            int_bits=self.int_bits_softmax,
            LUT_SIZE=self.lut_size_softmax,
            dim=-1,
            quant=self.quant,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.mat_mul_pv = QuantizedMatmul(
            in1_bits=self.nof_bits_matmul1,
            in2_bits=self.nof_bits_matmul2,
            quant=quant,
        )
        self.proj = QuantizedLinear(
            dim,
            dim,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.mat_mul_qk(q, k.transpose(-2, -1)) * self.scale
        attn = self.sf(attn)
        attn = self.attn_drop(attn)

        x = self.mat_mul_pv(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def set_quant(self):
        self.quant = True

    def unset_quant(self):
        self.quant = False
