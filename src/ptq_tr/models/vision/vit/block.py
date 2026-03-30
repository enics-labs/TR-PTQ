"""Transformer block implementation."""

import torch.nn as nn

from ptq_tr.models.vision.base import DropPath
from ptq_tr.models.vision.vit.attention import Attention
from ptq_tr.models.vision.vit.mlp import Mlp
from ptq_tr.quantization.modules.int_gelu import IntGeluTS
from ptq_tr.quantization.modules.q_layernorm import QLayerNorm
from ptq_tr.quantization.qparams import QauntParams


class Block(QauntParams):
    """Transformer Block: Multi-head Attention + MLP + LayerNorm."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        is_calibrate=False,
        act_layer=IntGeluTS,
        norm_layer=QLayerNorm,
        quant=None,
        quant_config=None,
    ):
        super().__init__(quant_config=quant_config)
        quant = self.quant if quant is None else quant
        self.quant = quant
        self.norm1 = norm_layer(
            embed_dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            quant=quant,
        )
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            quant=quant,
            quant_config=self,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(
            embed_dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            quant=quant,
        )

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            quant=quant,
            quant_config=self,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def set_quant(self):
        self.quant = True

    def unset_quant(self):
        self.quant = False
