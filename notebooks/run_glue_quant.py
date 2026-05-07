# -*- coding: utf-8 -*-
"""Generic GLUE quantized BERT runner for MRPC, CoLA, RTE, SST-2, QQP, MNLI, and QNLI."""

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, BertConfig, BertTokenizerFast


os.environ["HF_DATASETS_OFFLINE"] = "0"

PROJECT_DIR_CANDIDATES = [
    os.environ.get("GLUE_PROJECT_DIR"),
    os.environ.get("MRPC_PROJECT_DIR"),
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
        "Set GLUE_PROJECT_DIR or run this script from the project root."
    )

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mrcp_quant import (  # noqa: E402
    apply_experiment_config,
    apply_layer_quant_overrides,
    get_task_spec,
    load_experiment_config,
    nrmse,
    optimal_factor_search_lnorm,
    resolve_q_module_list,
    save_experiment_result,
    tokenize_task_batch,
    update_scale_factor,
)
from mrcp_quant.quantized_layers import (  # noqa: E402
    IntGeluTS,
    IntSoftmaxTS,
    QLayerNorm,
    QuantizedLinear,
    QuantizedMatmul,
    qHadamardProd,
)


def _config_path():
    return os.environ.get(
        "GLUE_QUANT_CONFIG",
        os.environ.get(
            "MRPC_QUANT_CONFIG",
            str(PROJECT_DIR / "notebooks" / "configs" / "quant_config.json"),
        ),
    )


def _move_inputs_to_device(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}


def _load_split(task, split, shuffle_seed=None):
    dataset = load_dataset(
        task.dataset_name,
        task.dataset_config,
        split=split,
        streaming=True,
    )
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    return dataset


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


def calibrate_model(model, task, tokenizer, q_module_list, config, device):
    calibration_config = config.get("calibration", {})
    num_samples = calibration_config.get("num_samples", 32)
    max_len = config.get("evaluation", {}).get("max_len", 128)

    if not _should_calibrate(q_module_list):
        print("Skipping calibration: q_module_list does not require observers.")
        return

    dataset = _load_split(task, "train")
    model.eval()

    with torch.no_grad():
        model.set_calibration_flag()
        print("Calibrating quantization parameters...")

        for i, sample in enumerate(dataset):
            if i == num_samples:
                break
            inputs = tokenize_task_batch(task, sample, tokenizer, max_length=max_len, padding=True)
            inputs = _move_inputs_to_device(inputs, device)
            print(i, end="|")
            _ = model(**inputs)

        model.unset_calibration_flag()
        print()
        print("Quantization parameters were set")


def _proxy_forward(model, task, tokenizer, sample, max_len, device):
    inputs = tokenize_task_batch(task, sample, tokenizer, max_length=max_len, padding="max_length")
    inputs = _move_inputs_to_device(inputs, device)
    with torch.no_grad():
        _ = model(**inputs)


def optimize_scale_factors(model, task, tokenizer, q_module_list, config, device):
    if QLayerNorm not in q_module_list:
        print("Skipping scale optimization: QLayerNorm is not quantized.")
        return

    scale_config = config.get("scale_optimization", {})
    num_samples = scale_config.get("num_samples", 30)
    if num_samples <= 0:
        print("Skipping scale optimization: num_samples <= 0.")
        return

    max_len = config.get("evaluation", {}).get("max_len", 128)
    dataset = _load_split(task, "train")

    with torch.no_grad():
        model.set_scale_opt()
        print("Optimizing model scale factors...")

        for i, sample in enumerate(dataset):
            if i == num_samples:
                break

            print(i, end="|")
            _proxy_forward(model, task, tokenizer, sample, max_len, device)

            for module in model.modules():
                if type(module) is not QLayerNorm or i != 0:
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


def evaluate_model(model, task, tokenizer, config, device):
    evaluation_config = config.get("evaluation", {})
    batch_size = evaluation_config.get("batch_size", 16)
    max_len = evaluation_config.get("max_len", 128)
    num_val_examples = evaluation_config.get("num_val_examples", task.default_num_val_examples)

    dataset = _load_split(task, task.validation_split, shuffle_seed=400)
    streamer = dataset.iter(batch_size=batch_size)
    metric = evaluate.load(task.metric_name, task.metric_config)

    predictions = []
    references = []
    losses = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(streamer, total=max(1, num_val_examples // batch_size))):
            if i * batch_size >= num_val_examples:
                break

            inputs = tokenize_task_batch(task, batch, tokenizer, max_length=max_len, padding="max_length")
            inputs = _move_inputs_to_device(inputs, device)
            labels = torch.tensor(batch["label"], device=device)

            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()

            predictions.extend(batch_preds)
            references.extend(batch["label"])
            if outputs.loss is not None:
                losses.append(outputs.loss.detach().cpu().item())

            if (i + 1) % 10 == 0:
                interim = metric.compute(predictions=predictions, references=references)
                print(f"\n[Batch {i + 1}] Interim {task.primary_metric}: {interim[task.primary_metric]:.4f}\n")

    metrics = metric.compute(predictions=predictions, references=references)
    average_loss = sum(losses) / len(losses) if losses else None
    return metrics, average_loss, len(references)


def main():
    config = load_experiment_config(_config_path())
    apply_experiment_config(config)

    task = get_task_spec(config.get("task_name", "mrpc"))
    model_name = config.get("model_name", task.default_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    hf_config = BertConfig.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model = task.model_class(hf_config)
    applied_layer_quant_overrides = apply_layer_quant_overrides(model, config)
    if applied_layer_quant_overrides:
        print("Applied layer quantization overrides:", applied_layer_quant_overrides)

    load_result = model.load_state_dict(hf_model.state_dict(), strict=False)
    print("Missing keys:", len(load_result.missing_keys))
    print("Unexpected keys:", len(load_result.unexpected_keys))
    print("Missing examples:", load_result.missing_keys[:30])

    model.to(device)

    q_module_list = resolve_q_module_list(config.get("q_module_list", ["QLayerNorm"]))
    resolved_q_module_names = q_module_names(q_module_list)
    model.set_q_module_list(q_module_list)
    model.set_quant()

    calibrate_model(model, task, tokenizer, q_module_list, config, device)
    optimize_scale_factors(model, task, tokenizer, q_module_list, config, device)
    metrics, average_loss, num_examples = evaluate_model(model, task, tokenizer, config, device)

    primary_metric_value = metrics[task.primary_metric]
    print(f"Final {task.primary_metric}: {primary_metric_value}")
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
            "task_name": task.name,
            "model_name": model_name,
            "quantized_module_paths": quantized_module_paths(model),
            "metrics": metrics,
            "primary_metric_name": task.primary_metric,
            "primary_metric_value": primary_metric_value,
            "num_val_examples": num_examples,
        },
    )
    print("Saved results:", result_path)


if __name__ == "__main__":
    main()
