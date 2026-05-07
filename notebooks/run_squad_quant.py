# -*- coding: utf-8 -*-
"""Configurable SQuAD quantized BERT runner."""

import os
import sys
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig, BertForQuestionAnswering


os.environ["HF_DATASETS_OFFLINE"] = "0"

PROJECT_DIR_CANDIDATES = [
    os.environ.get("SQUAD_PROJECT_DIR"),
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
        "Set SQUAD_PROJECT_DIR or run this script from the project root."
    )

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mrcp_quant import (  # noqa: E402
    CustomBertForQuestionAnswering,
    apply_experiment_config,
    apply_layer_quant_overrides,
    load_experiment_config,
    nrmse,
    optimal_factor_search_lnorm,
    resolve_q_module_list,
    save_experiment_result,
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
        "SQUAD_QUANT_CONFIG",
        str(PROJECT_DIR / "notebooks" / "configs" / "quant_config_squad.json"),
    )


def _move_inputs_to_device(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}


def _load_split(config, split, shuffle_seed=None):
    dataset = load_dataset(
        config.get("dataset_name", "squad"),
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


def tokenize_qa_batch(batch, tokenizer, max_len, stride, return_offsets_mapping=False):
    return tokenizer(
        batch["question"],
        batch["context"],
        truncation="only_second",
        max_length=max_len,
        stride=stride,
        return_offsets_mapping=return_offsets_mapping,
        padding="max_length",
        return_tensors="pt",
    )


def calibrate_model(model, tokenizer, q_module_list, config, device):
    calibration_config = config.get("calibration", {})
    evaluation_config = config.get("evaluation", {})
    num_samples = calibration_config.get("num_samples", 32)
    max_len = evaluation_config.get("max_len", 384)
    stride = evaluation_config.get("stride", 128)

    if not _should_calibrate(q_module_list):
        print("Skipping calibration: q_module_list does not require observers.")
        return

    dataset = _load_split(config, "train")
    model.eval()

    with torch.no_grad():
        model.set_calibration_flag()
        print("Calibrating quantization parameters...")

        for i, sample in enumerate(dataset):
            if i == num_samples:
                break
            inputs = tokenize_qa_batch(sample, tokenizer, max_len, stride)
            inputs = _move_inputs_to_device(inputs, device)
            print(i, end="|")
            _ = model(**inputs)

        model.unset_calibration_flag()
        print()
        print("Quantization parameters were set")


def _proxy_forward(model, tokenizer, sample, max_len, stride, device):
    inputs = tokenize_qa_batch(sample, tokenizer, max_len, stride)
    inputs = _move_inputs_to_device(inputs, device)
    with torch.no_grad():
        _ = model(**inputs)


def optimize_scale_factors(model, tokenizer, q_module_list, config, device):
    if QLayerNorm not in q_module_list:
        print("Skipping scale optimization: QLayerNorm is not quantized.")
        return

    scale_config = config.get("scale_optimization", {})
    evaluation_config = config.get("evaluation", {})
    num_samples = scale_config.get("num_samples", 5)
    if num_samples <= 0:
        print("Skipping scale optimization: num_samples <= 0.")
        return

    max_len = evaluation_config.get("max_len", 384)
    stride = evaluation_config.get("stride", 128)
    dataset = _load_split(config, "train")

    with torch.no_grad():
        model.set_scale_opt()
        print("Optimizing model scale factors...")

        for i, sample in enumerate(dataset):
            if i == num_samples:
                break

            print(i, end="|")
            _proxy_forward(model, tokenizer, sample, max_len, stride, device)

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


def _best_answer_for_sample(batch, encoded, outputs, sample_index):
    start_logits = outputs.start_logits[sample_index].detach().cpu().clone()
    end_logits = outputs.end_logits[sample_index].detach().cpu().clone()
    offsets = encoded["offset_mapping"][sample_index]
    sequence_ids = encoded.sequence_ids(sample_index)

    for token_index, sequence_id in enumerate(sequence_ids):
        if sequence_id != 1:
            start_logits[token_index] = -float("inf")
            end_logits[token_index] = -float("inf")

    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()
    if start_idx > end_idx:
        return ""

    start_char = offsets[start_idx][0].item()
    end_char = offsets[end_idx][1].item()
    if end_char <= start_char:
        return ""
    return batch["context"][sample_index][start_char:end_char]


def evaluate_model(model, tokenizer, config, device):
    evaluation_config = config.get("evaluation", {})
    batch_size = evaluation_config.get("batch_size", 16)
    max_len = evaluation_config.get("max_len", 384)
    stride = evaluation_config.get("stride", 128)
    num_val_examples = evaluation_config.get("num_val_examples", 10570)

    dataset = _load_split(config, "validation", shuffle_seed=400)
    streamer = dataset.iter(batch_size=batch_size)
    metric = evaluate.load("squad")

    predictions = []
    references = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(streamer, total=max(1, num_val_examples // batch_size))):
            if i * batch_size >= num_val_examples:
                break

            encoded = tokenize_qa_batch(
                batch,
                tokenizer,
                max_len,
                stride,
                return_offsets_mapping=True,
            )
            inputs = {
                key: value
                for key, value in encoded.items()
                if key != "offset_mapping"
            }
            outputs = model(**_move_inputs_to_device(inputs, device))

            for sample_index in range(len(batch["context"])):
                predictions.append({
                    "id": batch["id"][sample_index],
                    "prediction_text": _best_answer_for_sample(
                        batch,
                        encoded,
                        outputs,
                        sample_index,
                    ),
                })
                references.append({
                    "id": batch["id"][sample_index],
                    "answers": {
                        "text": batch["answers"][sample_index]["text"],
                        "answer_start": batch["answers"][sample_index]["answer_start"],
                    },
                })

            if (i + 1) % 10 == 0:
                interim = metric.compute(predictions=predictions, references=references)
                print(
                    f"\n[Batch {i + 1}] Interim exact_match: {interim['exact_match']:.2f} | "
                    f"f1: {interim['f1']:.2f}\n"
                )

    metrics = metric.compute(predictions=predictions, references=references)
    return metrics, len(references)


def main():
    config = load_experiment_config(_config_path())
    apply_experiment_config(config)

    model_name = config.get("model_name", "bert-large-uncased-whole-word-masking-finetuned-squad")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    hf_config = BertConfig.from_pretrained(model_name)
    hf_model = BertForQuestionAnswering.from_pretrained(model_name)

    model = CustomBertForQuestionAnswering(hf_config)
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

    calibrate_model(model, tokenizer, q_module_list, config, device)
    optimize_scale_factors(model, tokenizer, q_module_list, config, device)
    metrics, num_examples = evaluate_model(model, tokenizer, config, device)

    primary_metric_name = "f1"
    primary_metric_value = metrics[primary_metric_name]
    print("Final exact_match:", metrics["exact_match"])
    print("Final f1:", primary_metric_value)

    output_config = dict(config)
    output_config["q_module_list"] = resolved_q_module_names

    result_path = save_experiment_result(
        accuracy=primary_metric_value,
        loss=None,
        configuration=output_config,
        quantized=resolved_q_module_names,
        output_dir=PROJECT_DIR / "output",
        extra={
            "task_name": "squad",
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
