"""Patch embedding implementation."""

import torch.nn as nn

from ptq_tr.quantization.qparams import QauntParams

try:
    from timm.models.layers import to_2tuple
except ImportError:
    def to_2tuple(x):
        return (x, x) if not isinstance(x, tuple) else x


class PatchEmbed(QauntParams):
    r"""Image to Patch Embedding."""

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        quant=None,
        quant_config=None,
        norm_layer=None,
    ):
        super().__init__(quant_config=quant_config)
        quant = self.quant if quant is None else quant
        self.quant = quant
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(
                embed_dim,
                in1_bits=self.nof_bits_lnorm1,
                in2_bits=self.nof_bits_lnorm2,
                quant=quant,
            )
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
