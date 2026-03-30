"""Model factory functions for vision models."""

import torch

from ptq_tr.models.vision.vit.deit.model import DistilledVisionTransformer
from ptq_tr.models.vision.vit.swin.model import SwinTransformer
from ptq_tr.quantization.modules.q_layernorm import QLayerNorm


def deit_base_patch16_224(pretrained=False, quant=False, q_module_list=[], **kwargs):
    model = DistilledVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        quant=quant,
        q_module_list=q_module_list,
        distilled=False,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth", map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


def deit_small_patch16_224(pretrained=False, quant=False, q_module_list=[], **kwargs):
    model = DistilledVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        q_module_list=q_module_list,
        quant=quant,
        distilled=False,
        **kwargs,
    )

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth", map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


def deit_tiny_patch16_224(pretrained=False, quant=False, q_module_list=[], **kwargs):
    model = DistilledVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        q_module_list=q_module_list,
        quant=quant,
        distilled=False,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth", map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


def swin_tiny_patch4_window7_224(pretrained=False, quant=False, q_module_list=[], **kwargs):
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        quant=quant,
        norm_layer=QLayerNorm,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        q_module_list=q_module_list,
        **kwargs,
    )

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


def swin_small_patch4_window7_224(pretrained=False, quant=False, q_module_list=[], **kwargs):
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        norm_layer=QLayerNorm,
        embed_dim=96,
        quant=quant,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        q_module_list=q_module_list,
        **kwargs,
    )

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model


def swin_base_patch4_window7_224(pretrained=False, quant=False, q_module_list=[], **kwargs):
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        norm_layer=QLayerNorm,
        quant=quant,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        q_module_list=q_module_list,
        **kwargs,
    )

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    return model
