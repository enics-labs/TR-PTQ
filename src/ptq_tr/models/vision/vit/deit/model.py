"""DeiT model implementation."""

import torch
import torch.nn as nn

from ptq_tr.models.vision.base import QuantTransformer
from ptq_tr.models.vision.vit.block import Block
from ptq_tr.quantization.modules.int_gelu import IntGeluTS
from ptq_tr.quantization.modules.q_hadamard import qHadamardProd
from ptq_tr.quantization.modules.q_layernorm import QLayerNorm
from ptq_tr.quantization.modules.quant_linear import QuantizedLinear
from ptq_tr.quantization.modules.quant_matmul import QuantizedMatmul


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class DistilledVisionTransformer(QuantTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        distilled=True,
        quant=False,
        is_calibrate=False,
        q_module_list=[QuantizedLinear, QuantizedMatmul, qHadamardProd, QLayerNorm],
        **kwargs,
    ):
        super().__init__(
            quant=quant,
            is_calibrate=is_calibrate,
            q_module_list=q_module_list,
        )
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    is_calibrate=self.is_calibrate,
                    act_layer=IntGeluTS,
                    norm_layer=QLayerNorm,
                    quant=self.quant,
                    **kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.norm = QLayerNorm(
            embed_dim,
            in1_bits=self.nof_bits_lnorm1,
            in2_bits=self.nof_bits_lnorm2,
            quant=self.quant,
        )

        self.head = (
            QuantizedLinear(
                embed_dim,
                num_classes,
                nof_bits1=8,
                nof_bits2=8,
                quant=self.quant,
            )
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = (
            QuantizedLinear(
                embed_dim,
                num_classes,
                nof_bits1=8,
                nof_bits2=8,
                quant=self.quant,
            )
            if distilled
            else None
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, QuantizedLinear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None or isinstance(m, QuantizedLinear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, QLayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.dist_token is None:
            return x[:, 0]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x_cls, x_dist = x
            x = self.head(x_cls), self.head_dist(x_dist)
            if not self.training:
                return (x[0] + x[1]) / 2
            else:
                return x
        else:
            x = self.head(x)
        return x
