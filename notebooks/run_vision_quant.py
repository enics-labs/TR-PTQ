# -*- coding: utf-8 -*-
"""Configurable DeiT/Swin quantized ImageNet runner."""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


PROJECT_DIR_CANDIDATES = [
    os.environ.get("VISION_PROJECT_DIR"),
    Path.cwd(),
    Path("/content/mrcp-tr-ptq"),
]

for candidate in PROJECT_DIR_CANDIDATES:
    if not candidate:
        continue
    candidate = Path(candidate).expanduser()
    if (candidate / "mrcp_quant").is_dir():
        PROJECT_DIR = candidate
        break
else:
    raise FileNotFoundError(
        "Could not find the project folder containing mrcp_quant. "
        "Set VISION_PROJECT_DIR or run this script from the project root."
    )

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mrcp_quant import (  # noqa: E402
    IntGeluTS,
    IntSoftmaxTS,
    QLayerNorm,
    QuantizedLinear,
    QuantizedMatmul,
    apply_experiment_config,
    apply_layer_quant_overrides,
    get_vision_model_loader,
    load_experiment_config,
    nrmse,
    optimal_factor_search_lnorm,
    qHadamardProd,
    resolve_q_module_list,
    save_experiment_result,
    update_scale_factor,
)


def _config_path():
    return os.environ.get(
        "VISION_QUANT_CONFIG",
        str(PROJECT_DIR / "notebooks" / "configs" / "quant_config_vision.json"),
    )


def _should_calibrate(q_module_list):
    return any(
        module_class in q_module_list
        for module_class in (
            QLayerNorm,
            IntSoftmaxTS,
            QuantizedLinear,
            QuantizedMatmul,
            IntGeluTS,
            qHadamardProd,
        )
    )


def q_module_names(q_module_list):
    return [module_class.__name__ for module_class in q_module_list]


def quantized_module_paths(model):
    return [
        {"path": name, "type": type(module).__name__}
        for name, module in model.named_modules()
        if getattr(module, "quant", False) is True
    ]


def build_preprocess(config):
    eval_config = config.get("evaluation", {})
    resize_size = eval_config.get("resize_size", 256)
    image_size = eval_config.get("image_size", 224)
    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_split(config, split, shuffle_seed=None):
    dataset_config = config.get("dataset", {})
    kwargs = {
        "split": split,
        "streaming": True,
    }
    if dataset_config.get("token", True):
        kwargs["token"] = True
    if dataset_config.get("trust_remote_code", True):
        kwargs["trust_remote_code"] = True
    dataset = load_dataset(dataset_config.get("name", "imagenet-1k"), **kwargs)
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    return dataset


def _image_to_tensor(image, preprocess):
    if isinstance(image, dict) and "bytes" in image:
        import io

        image = Image.open(io.BytesIO(image["bytes"]))
    return preprocess(image)


def _batch_to_device(batch, preprocess, device):
    images = [_image_to_tensor(image, preprocess) for image in batch["image"]]
    labels = torch.tensor(batch["label"], device=device, dtype=torch.long)
    return torch.stack(images).to(device), labels


def calibrate_model(model, q_module_list, config, preprocess, device):
    calibration_config = config.get("calibration", {})
    num_samples = calibration_config.get("num_samples", 32)
    if not _should_calibrate(q_module_list):
        print("Skipping calibration: q_module_list does not require observers.")
        return
    if num_samples <= 0:
        print("Skipping calibration: num_samples <= 0.")
        return

    dataset_config = config.get("dataset", {})
    dataset = _load_split(config, dataset_config.get("train_split", "train"))
    model.eval()

    with torch.no_grad():
        model.set_calibration_flag()
        print("Calibrating quantization parameters...")
        for index, sample in enumerate(dataset):
            if index == num_samples:
                break
            image = _image_to_tensor(sample["image"], preprocess).unsqueeze(0).to(device)
            print(index, end="|")
            _ = model(image)
        model.unset_calibration_flag()
        print()
        print("Quantization parameters were set")


def optimize_scale_factors(model, q_module_list, config, preprocess, device):
    if QLayerNorm not in q_module_list:
        print("Skipping scale optimization: QLayerNorm is not quantized.")
        return

    scale_config = config.get("scale_optimization", {})
    num_samples = scale_config.get("num_samples", 0)
    if num_samples <= 0:
        print("Skipping scale optimization: num_samples <= 0.")
        return

    dataset_config = config.get("dataset", {})
    dataset = _load_split(config, dataset_config.get("train_split", "train"), shuffle_seed=12)

    with torch.no_grad():
        model.set_scale_opt()
        print("Optimizing model scale factors...")
        for index, sample in enumerate(dataset):
            if index == num_samples:
                break
            print(index, end="|")
            image = _image_to_tensor(sample["image"], preprocess).unsqueeze(0).to(device)
            _ = model(image)
            for module in model.modules():
                if type(module) is not QLayerNorm or index != 0:
                    continue
                _ = module.forward_pass(module.opt_input)
                weight_error = nrmse(
                    module.w_obs.dequantizer(module.weight_integer),
                    module.weight,
                )
                print(f"____________rmse:{weight_error}_____________")
                if weight_error > 0.03:
                    scale = optimal_factor_search_lnorm(
                        module,
                        module.w_obs,
                        is_weight=False,
                        alpha=0.5,
                        betta=2.0,
                        itr=50,
                    )
                    print("search_scale_w", scale)
                    update_scale_factor(
                        module,
                        module.w_obs,
                        module.w_obs.min_val,
                        module.w_obs.max_val,
                        scale,
                    )
        model.unset_scale_opt()
        print()


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size).item())
    return result


def evaluate_model(model, config, preprocess, device):
    evaluation_config = config.get("evaluation", {})
    dataset_config = config.get("dataset", {})
    batch_size = evaluation_config.get("batch_size", 8)
    num_val_examples = evaluation_config.get("num_val_examples", 50000)
    topk = tuple(evaluation_config.get("topk", [1, 5]))
    dataset = _load_split(
        config,
        dataset_config.get("validation_split", "validation"),
        shuffle_seed=400,
    )
    streamer = dataset.iter(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()

    top_totals = {k: 0.0 for k in topk}
    losses = []
    seen = 0
    model.eval()

    with torch.no_grad():
        for index, batch in enumerate(tqdm(streamer, total=max(1, num_val_examples // batch_size))):
            if seen >= num_val_examples:
                break
            images, labels = _batch_to_device(batch, preprocess, device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            current_batch_size = labels.size(0)
            seen += current_batch_size
            losses.append(loss.detach().cpu().item())
            for k, value in zip(topk, accuracy(outputs, labels, topk=topk)):
                top_totals[k] += value * current_batch_size
            if (index + 1) % 10 == 0:
                top1 = top_totals.get(1, 0.0) / max(seen, 1)
                print(f"\n[Batch {index + 1}] Interim top1: {top1:.2f}\n")

    metrics = {f"top{k}": top_totals[k] / max(seen, 1) for k in topk}
    average_loss = sum(losses) / len(losses) if losses else None
    return metrics, average_loss, seen


def main():
    config = load_experiment_config(_config_path())
    apply_experiment_config(config)

    model_name = config.get("model_name", "deit_tiny_patch16_224")
    pretrained = config.get("pretrained", True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = build_preprocess(config)

    q_module_list = resolve_q_module_list(config.get("q_module_list", ["QLayerNorm"]))
    resolved_q_module_names = q_module_names(q_module_list)
    loader = get_vision_model_loader(model_name)
    model = loader(pretrained=pretrained, q_module_list=q_module_list, quant=False)
    applied_layer_quant_overrides = apply_layer_quant_overrides(model, config)
    if applied_layer_quant_overrides:
        print("Applied layer quantization overrides:", applied_layer_quant_overrides)

    model.to(device)
    model.set_q_module_list(q_module_list)
    model.set_quant()

    calibrate_model(model, q_module_list, config, preprocess, device)
    optimize_scale_factors(model, q_module_list, config, preprocess, device)
    metrics, average_loss, num_examples = evaluate_model(model, config, preprocess, device)

    primary_metric_name = "top1"
    primary_metric_value = metrics.get(primary_metric_name)
    print("Final top1:", primary_metric_value)
    if "top5" in metrics:
        print("Final top5:", metrics["top5"])
    print("Final Loss:", average_loss)

    output_config = dict(config)
    output_config["q_module_list"] = resolved_q_module_names
    result_path = save_experiment_result(
        accuracy=primary_metric_value,
        loss=average_loss,
        configuration=output_config,
        quantized=resolved_q_module_names,
        output_dir=PROJECT_DIR / "output",
        extra={
            "task_name": "imagenet",
            "model_name": model_name,
            "quantized_module_paths": quantized_module_paths(model),
            "metrics": metrics,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "num_val_examples": num_examples,
        },
    )
    print("Saved results:", result_path)


if __name__ == "__main__":
    main()
