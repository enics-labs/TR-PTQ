"""Window attention implementation."""

import torch
import torch.nn as nn

from ptq_tr.quantization.modules.int_softmax import IntSoftmaxTS
from ptq_tr.quantization.modules.quant_linear import QuantizedLinear
from ptq_tr.quantization.modules.quant_matmul import QuantizedMatmul
from ptq_tr.quantization.qparams import QauntParams

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_


class WindowAttention(QauntParams):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        quant=False,
        proj_drop=0.0,
    ):
        super().__init__()
        self.quant = quant
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = QuantizedLinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            nof_bits1=8,
            nof_bits2=8,
            quant=quant,
        )

        self.mat_mul_qk = QuantizedMatmul(in1_bits=8, in2_bits=8, quant=quant)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mat_mul_pv = QuantizedMatmul(in1_bits=8, in2_bits=8, quant=quant)
        self.proj = QuantizedLinear(dim, dim, nof_bits1=8, nof_bits2=8, quant=quant)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.softmax = IntSoftmaxTS(nof_bits=8, LUT_SIZE=16, dim=-1, quant=quant)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.mat_mul_qk(q, k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = self.mat_mul_pv(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops
