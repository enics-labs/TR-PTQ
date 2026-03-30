"""Shared helpers for vision workflow entrypoints."""

from ptq_tr.common.device import get_default_device
from ptq_tr.dataloaders.vision import build_vision_dataloaders
from ptq_tr.models.vision import (
    deit_base_patch16_224,
    deit_small_patch16_224,
    deit_tiny_patch16_224,
    swin_base_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_tiny_patch4_window7_224,
)
from ptq_tr.quantization.modules import (
    IntGeluTS,
    IntSoftmaxTS,
    QLayerNorm,
    QuantizedLinear,
    QuantizedMatmul,
    qHadamardProd,
)


VISION_MODEL_REGISTRY = {
    "deit_tiny_patch16_224": deit_tiny_patch16_224,
    "deit_small_patch16_224": deit_small_patch16_224,
    "deit_base_patch16_224": deit_base_patch16_224,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
}

VISION_Q_MODULE_REGISTRY = {
    "QuantizedLinear": QuantizedLinear,
    "QuantizedMatmul": QuantizedMatmul,
    "IntSoftmaxTS": IntSoftmaxTS,
    "QLayerNorm": QLayerNorm,
    "IntGeluTS": IntGeluTS,
    "qHadamardProd": qHadamardProd,
}


def resolve_q_module_list(q_module_names):
    if not q_module_names:
        return []

    if len(q_module_names) == 1 and q_module_names[0].lower() == "none":
        return []

    unknown = [name for name in q_module_names if name not in VISION_Q_MODULE_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown q modules: {unknown}. Available: {list(VISION_Q_MODULE_REGISTRY)}")

    return [VISION_Q_MODULE_REGISTRY[name] for name in q_module_names]


def build_vision_runtime(
    *,
    model_name="deit_tiny_patch16_224",
    dataset_name="imagenet",
    batch_size=8,
    calib_batch_size=1,
    scale_opt_batch_size=1,
    pretrained=True,
    quant=False,
    q_module_names=("QLayerNorm",),
    hf_token=None,
    device=None,
):
    if model_name not in VISION_MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(VISION_MODEL_REGISTRY)}")

    q_module_list = resolve_q_module_list(list(q_module_names))
    model = VISION_MODEL_REGISTRY[model_name](
        pretrained=pretrained,
        q_module_list=q_module_list,
        quant=quant,
    )

    device = device or get_default_device()
    model = model.to(device)
    model.eval()

    if q_module_list:
        model.set_quant()

    dataloaders = build_vision_dataloaders(
        dataset_name,
        batch_size=batch_size,
        calib_batch_size=calib_batch_size,
        scale_opt_batch_size=scale_opt_batch_size,
        token=hf_token if hf_token is not None else True,
    )

    return {
        "model": model,
        "device": device,
        "q_module_list": q_module_list,
        "val_loader": dataloaders["val_loader"],
        "train_loader": dataloaders["train_loader"],
        "scale_opt_loader": dataloaders["scale_opt_loader"],
        "single_image_iter": dataloaders["single_image_iter"],
        "datasets": dataloaders["datasets"],
    }
