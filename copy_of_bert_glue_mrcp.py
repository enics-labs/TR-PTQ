# -*- coding: utf-8 -*-
"""MRPC quantized BERT experiment entry point."""

import os
import sys
from pathlib import Path
os.environ["HF_DATASETS_OFFLINE"] = "0"

PROJECT_DIR_CANDIDATES = [
    os.environ.get("MRPC_PROJECT_DIR"),
    Path.cwd(),
    Path("/content/mrcp-tr-ptq"),
    Path("/content/drive/MyDrive/mrcp-tr-ptq"),
    Path("/content/drive/MyDrive/Colab Notebooks/mrcp-tr-ptq"),
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
        "In Colab, mount Drive and set MRPC_PROJECT_DIR to your project path, "
        "for example: os.environ['MRPC_PROJECT_DIR'] = "
        "'/content/drive/MyDrive/mrcp-tr-ptq'."
    )

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mrcp_quant import *

CONFIG_PATH = os.environ.get("MRPC_QUANT_CONFIG", str(PROJECT_DIR / "quant_config.json"))
experiment_config = load_experiment_config(CONFIG_PATH)
apply_experiment_config(experiment_config)

"""# Model Initialization"""

from transformers import (
    BertTokenizerFast,
    BertConfig,
    AutoModelForSequenceClassification

)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = experiment_config.get("model_name", "lrs21/bert-base-uncased-finetuned-glue-mrpc")

if (
    globals().get("load_mem") != 1
    or globals().get("loaded_model_name") != model_name
    or "tokenizer" not in globals()
    or "hf_config" not in globals()
    or "hf_model" not in globals()
):

  # Tokenizer & config
  tokenizer = BertTokenizerFast.from_pretrained(model_name)
  hf_config = BertConfig.from_pretrained(model_name)

  # HuggingFace reference model
  hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
  load_mem = 1
  loaded_model_name = model_name

# Your custom quantized model
model = CustomBertForMRPC(hf_config)
applied_layer_quant_overrides = apply_layer_quant_overrides(model, experiment_config)
if applied_layer_quant_overrides:
    print("Applied layer quantization overrides:", applied_layer_quant_overrides)

# Load weights
res = model.load_state_dict(hf_model.state_dict(), strict=False)
print("Missing keys:", len(res.missing_keys))
print("Unexpected keys:", len(res.unexpected_keys))
print("Missing examples:", res.missing_keys[:30])


model.to(device)

"""## quant model"""

q_module_list = resolve_q_module_list(
    experiment_config.get("q_module_list", ["QLayerNorm"])
)

model.set_q_module_list(q_module_list)
model.set_quant()

def audit_modes(model):
    note = []
    for name, m in model.named_modules():
        q = getattr(m, "quant", None)
        # cal = getattr(m, "is_calibrate", None)
        opt = getattr(m, "is_opt_scale", None)
        if (q is True) or (opt is True):
            note.append((name, type(m).__name__, q, opt))
    return note

note = audit_modes(model)
print("modules not in pure-float mode:", len(note))
print(*note[:50], sep="\n")

"""# Calibration"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# !pip uninstall -y tokenizers
# # !pip install "tokenizers>=0.20,<0.21"
# !pip install tokenizers==0.20.3

# !pip install -U transformers datasets huggingface_hub

from datasets import load_dataset
import torch

dataset = load_dataset(
    "nyu-mll/glue",
    "mrpc",
    split="train",
    streaming=True
)

# Use the streaming GLUE-CoLA dataset to calibrate the quantized model

# q_module_list = [QLayerNorm, IntSoftmaxTS, QuantizedLinear, QuantizedMatmul]
calibration_config = experiment_config.get("calibration", {})
calibration_num_samples = calibration_config.get("num_samples", 32)

if (
    QuantizedLinear in q_module_list
    or QuantizedMatmul in q_module_list
    or IntSoftmaxTS in q_module_list
    or QLayerNorm in q_module_list
):
    with torch.no_grad():
        model.set_calibration_flag()
        print()

        for i, sample in enumerate(dataset):
            if i == calibration_num_samples:
                break

            # GLUE-MRCP has a single sentence
            s1 = sample["sentence1"]
            s2 = sample["sentence2"]
            label = sample["label"]  # 0/1

            inputs = tokenizer(
                s1,
                s2,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)

            print(i, end="|")
            _ = model(**inputs)

        model.unset_calibration_flag()
        print()
        print("Quantization parameters were set")

"""## Optimizing input scale"""

# before optimization

def cosine_similarity(tensor1, tensor2, eps=1e-8):
    """Calculates cosine similarity between two tensors."""
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    dot_product = torch.dot(tensor1, tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    return dot_product / (norm1 * norm2 + eps)

def nrmse(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8, norm: str = "var") -> torch.Tensor:
    """
    Compute Normalized Root Mean Squared Error (NRMSE).

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor (same shape as y_true).
        eps: Small constant to avoid division by zero.
        norm: Normalization method:
              - "var": normalize by sqrt(Var(y_true))  (default, scale-invariant)
              - "minmax": normalize by (max(y_true) - min(y_true))
              - "mean": normalize by mean(|y_true|)
              - "none": no normalization (just RMSE)

    Returns:
        Scalar tensor with NRMSE value.
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)

    if norm == "var":
        denom = torch.sqrt(y_true.var(unbiased=False) + eps)
    elif norm == "minmax":
        denom = (y_true.max() - y_true.min()).clamp(min=eps)
    elif norm == "mean":
        denom = y_true.abs().mean().clamp(min=eps)
    elif norm == "none":
        denom = 1.0
    else:
        raise ValueError(f"Unknown normalization method: {norm}")

    return rmse / denom

import numpy as np

def calc_cosine(layer, original_output, x, alpha=0.9, beta=1.1, num_scales=10):
    """
    Optimizes the scale factors for a layer normalization module using cosine similarity.

    Args:
        layer: The LayerNorm module whose scale factors will be optimized.
        original_output: A reference output for comparison.
        x: Input tensor to compute the cosine similarity.
        alpha: Lower bound for scale factor search.
        beta: Upper bound for scale factor search.
        num_scales: Number of scale factors to test within the [alpha, beta] range.

    Returns:
        None. Updates the layer with the optimal scale factors for weights and inputs.
    """
    # Optimize weight scale factor
    optimal_weight_scale = find_optimal_scale_factor(layer.update_w_scale_factor,
                                                     layer.forward_pass,
                                                     x,
                                                     original_output,
                                                     alpha, beta, num_scales)

    # Optimize input scale factor
    optimal_in_scale = find_optimal_scale_factor(layer.update_in_scale_factor,
                                                 layer.forward_pass,
                                                 x,
                                                 original_output,
                                                 alpha, beta, num_scales)

    # Apply the optimal scale factors to layer
    layer.set_optimal_w_scale_factor(optimal_weight_scale)
    layer.set_optimal_in_scale_factor(optimal_in_scale)


def find_optimal_scale_factor(update_scale_fn, forward_pass, x, original_output, alpha, beta, num_scales):
    """
    Finds the optimal scale factor that maximizes cosine similarity.

    Args:
        update_scale_fn: Function to update the scale factor.
        forward_pass: The forward pass function of the layer being optimized.
        x: Input tensor.
        original_output: The reference output for comparison.
        alpha: Lower bound for scale factor search.
        beta: Upper bound for scale factor search.
        num_scales: Number of scale factors to test.

    Returns:
        optimal_scale (float): The scale factor that gives the highest cosine similarity.
    """
    best_cosine = -float('inf')
    optimal_scale = 1.0

    # Test scale factors within the range [alpha, beta]
    for scale_factor in np.linspace(alpha, beta, num=num_scales):
        update_scale_fn(scale_factor)
        approx_output = forward_pass(x)
        cosine_sim = cosine_similarity(original_output, approx_output)

        if cosine_sim > best_cosine:
            best_cosine = cosine_sim
            optimal_scale = scale_factor

    return optimal_scale

def update_scale_factor(mdl, obs, cur_min, cur_max, search_scale):
      obs.min_val = cur_min * search_scale
      obs.max_val = cur_max * search_scale
      obs.calculate_qparams()
      mdl.is_weights_quantized = False


def optimal_factor_search(x, mdl, obs, is_weight=False, alpha=0.5, betta=2.0, itr=10, cosine_similarity=cosine_similarity):

  # sample the original scale parameters
  cur_min = obs.min_val
  cur_max = obs.max_val

  # base cosine and the optimal scale
  optimal_cosine = -1
  optimal_scale  = 1

  for search_scale in np.linspace(alpha, betta, num=itr):

      update_scale_factor(mdl, obs, cur_min, cur_max, search_scale)

      # calc cosine similarity
      approx_output = mdl.forward_pass(x)
      original_output = mdl.float_forward_pass(x)
      cosine_sim = cosine_similarity(original_output, approx_output)

      if cosine_sim > optimal_cosine:
          optimal_cosine = cosine_sim
          optimal_scale = search_scale

  # revert changes
  update_scale_factor(mdl, obs, cur_min, cur_max, 1)

  return optimal_scale

def optimal_factor_search2(x1, x2, mdl, obs, is_weight=False, alpha=0.5, betta=2.0, itr=10):

  # sample the original scale parameters
  cur_min = obs.min_val
  cur_max = obs.max_val

  # base cosine and the optimal scale
  optimal_cosine = -1
  optimal_scale  = 1

  for search_scale in np.linspace(alpha, betta, num=itr):

      update_scale_factor(mdl, obs, cur_min, cur_max, search_scale)

      # calc cosine similarity
      approx_output = mdl.forward_pass(x1, x2)
      original_output = mdl.float_forward_pass(x1, x2)
      cosine_sim = cosine_similarity(original_output, approx_output)

      if cosine_sim > optimal_cosine:
          optimal_cosine = cosine_sim
          optimal_scale = search_scale

  # revert changes
  update_scale_factor(mdl, obs, cur_min, cur_max, 1)

  return optimal_scale

def optimal_factor_search_lnorm(mdl, obs, is_weight=False, alpha=0.5, betta=2.0, itr=10):

  # sample the original scale parameters
  cur_min = obs.min_val
  cur_max = obs.max_val

  # base cosine and the optimal scale
  optimal_cosine = 100
  optimal_scale  = 1

  print("_______________________________________________")
  for search_scale in np.linspace(alpha, betta, num=itr):

      update_scale_factor(mdl, obs, cur_min, cur_max, search_scale)
      weight_integer = obs.quantizer(mdl.weight)

      # calc cosine similarity
      cosine_sim = nrmse(obs.dequantizer(weight_integer), mdl.weight)

      print(f"search_scale:{search_scale}, optimal_cosine:{optimal_cosine}, cosine_sim:{cosine_sim}")
      if (optimal_cosine - cosine_sim) > 0.1:
          optimal_cosine = cosine_sim
          optimal_scale = search_scale
  print("_______________________________________________")

  # revert changes
  update_scale_factor(mdl, obs, cur_min, cur_max, 1)
  print("-----------------")
  return optimal_scale

def insert_proxi_opt(batch, tokenizer):
    inputs = tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Setup ---
sns.set(style="whitegrid", font_scale=1.4)

# Store matrices by layer index
sampled_layers = {}
plotting_config = experiment_config.get("plotting", {})
MAX_SAMPLES = plotting_config.get("max_samples", 12)
PLOT_SAMPLE_BATCHES = plotting_config.get("sample_batches", 3)

# --- Sampling function ---
def sample_layers(var_name="weight"):
    global sampled_layers

    sampled_layers.clear()

    for i, sample in enumerate(dataset):
        if i == PLOT_SAMPLE_BATCHES:
            break

        for j, m in enumerate(model.modules()):
            if isinstance(m, QLayerNorm):
                if j not in sampled_layers and len(sampled_layers) < MAX_SAMPLES:
                    if var_name == "weight":
                        sampled_layers[j] = m.w_obs.dequantizer(m.weight_integer).detach().clone().cpu()
                        # sampled_layers[j] = m.weight.detach().clone().cpu()
                    # elif var_name == "activ":
                    #     norm = "var"
                    #     _ , zero_point_input_normed = m.in_obs_normalize.calculate_qparams()
                    #     # sampled_layers[j] = (m.in_obs_normalize(m.x_test) - zero_point_input_normed.int()).detach().clone().cpu()
                    #     print(f"---------------------- j {j} ------------------------------------")
                    #     gbl_nrmse = cosine_similarity(
                    #         m.in_obs_normalize.dequantizer(
                    #         m.in_obs_normalize.quantizer(
                    #             m.x_test
                    #             )
                    #         ),
                    #           m.x_test
                    #     )
                    #     top_sc = 1

                    #     for sc in np.linspace(0.5, 2, 50):
                    #         # print(f"[j: {j}; sc: {sc}]",
                    #         lcl_nrmse = cosine_similarity(
                    #             m.in_obs_normalize.dequantizer(
                    #             m.in_obs_normalize.quantizer(
                    #                 m.x_test*sc
                    #                 )
                    #             ) / sc,
                    #               m.x_test)
                    #         # print(f"sc:{sc}, lcl_nrmse:{lcl_nrmse}")

                    #         if lcl_nrmse > gbl_nrmse:
                    #             top_sc = sc
                    #             gbl_nrmse = lcl_nrmse

                    #     xq = m.in_obs_normalize.quantizer(
                    #             m.x_test*top_sc
                    #             )
                    #     xq_var = xq.float().var(dim=-1, unbiased=False, keepdim=True) * (m.scale_input_normed) ** 2 / top_sc
                    #     x_var = m.x_test.var(dim=-1, unbiased=False, keepdim=True)
                    #     print("xq_var", xq_var)
                    #     print("x_var", x_var)

                    #     print(f"top_sc:{top_sc}, gbl_nrmse:{gbl_nrmse}")
                    #     sampled_layers[j] = (m.in_obs_normalize.quantizer(m.x_test*top_sc)).detach().clone().cpu()

                    # else:
                    #     # _ , zero_point_input_normed = m.in_obs_normalize.calculate_qparams()
                    #     # mean_val = m.in_obs_normalize(m.x_test) - zero_point_input_normed
                    #     # mean_val = m.betta_gamma_forward_pass(m.x_test)
                    #     mean_val = m.x_test

                    #     # _ , zero_point_input = m.in_obs.calculate_qparams()
                    #     # x_q = m.in_obs.quantizer(mean_val)  # Quantize the input
                    #     # after_norm = x_q - zero_point_input.int()  # Adjust input by zero point

                    #     # sampled_layers[j] = after_norm.detach().clone().cpu()
                    #     sampled_layers[j] = mean_val.clone().cpu()
                    #     # print(sampled_layers[j], sampled_layers[j].shape)


        break

    print(f"✅ Sampled {len(sampled_layers)} QLayerNorm.weight_integer matrices")

# --- Plotting function (ICML-style) ---
def plot_distributions_icml(bins=200, normalize=False, after_layers=None):
    n = len(sampled_layers)
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(6.5 * cols, 4.5 * rows))

    for idx, (j, weights) in enumerate(sampled_layers.items()):
        plt.subplot(rows, cols, idx + 1)
        original = weights.view(-1).float().numpy()

        if normalize and after_layers and j in after_layers:
            modified = after_layers[j].view(-1).float().numpy()
            sns.histplot(modified, bins=bins, kde=True, color="crimson", label="After", stat="density", element="step")
            sns.histplot(original, bins=bins, kde=True, color="skyblue", label="Before", stat="density", element="step")
            plt.legend()
        else:
            sns.histplot(original, bins=bins, kde=True, color="navy", stat="density", element="step")

        plt.title(f"Layer {j}", fontsize=16)
        plt.xlabel("Weight Value", fontsize=14)
        plt.ylabel("Density", fontsize=14)

    plt.tight_layout()
    if normalize:
        plt.suptitle("Weight Distribution Before vs After Normalization", fontsize=18, y=1.02)
    else:
        plt.suptitle("QLayerNorm Weight Distributions", fontsize=18, y=1.02)

    plt.show()

import copy

# Deep copy for tokenizer (class instance)
tokenizer_copy = copy.deepcopy(tokenizer)

# Deep copy for sample (dict of strings & int)
sample_copy = copy.deepcopy(sample)

insert_proxi_opt(sample_copy, tokenizer_copy)

sampled_layers = {}
sample_layers(var_name="weight")
plot_distributions_icml()

# sampled_layers = {}

# sample_layers(var_name="else")
# plot_distributions_icml()

print("optimizing model scale-factors...")

scale_optimization_config = experiment_config.get("scale_optimization", {})
scale_optimization_num_samples = scale_optimization_config.get("num_samples", 30)

with torch.no_grad():

    model.set_scale_opt()
    print()

    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    for i, sample in enumerate(dataset):
        if i == scale_optimization_num_samples:
            break

        print(i, end="|")

        # ---- CoLA proxy forward ----
        insert_proxi_opt(sample, tokenizer)

        for m in model.modules():
            if type(m) not in q_module_list:
                continue

            # if type(m) in [QuantizedLinear]:
            #     x = m.opt_input
            #     dq_output = m.forward_pass(x)
            #     output = m.float_forward_pass(x)
            #     cs = cosine_similarity(output, dq_output)
            #     if cs < (1 - 0.01):
            #         print("---------------")
            #         print("QuantizedLinear: cosine similarity:", f"{float(cs):.2f}")
            #         search_scale_w = optimal_factor_search(x, m, m.w_obs, is_weight=True, alpha=0.5, betta=2.0, itr=50)
            #         update_scale_factor(m, m.w_obs, m.w_obs.min_val, m.w_obs.max_val,    search_scale_w)
            #         search_scale_in = optimal_factor_search(x, m, m.in_obs, is_weight=True, alpha=0.5, betta=2.0, itr=50)
            #         update_scale_factor(m, m.in_obs, m.in_obs.min_val, m.in_obs.max_val, search_scale_in)

            # ---- LayerNorm scale optimization ----
            if type(m) is QLayerNorm and i == 0:
                x = m.opt_input

                dq_output = m.forward_pass(x)
                cs = nrmse(
                    m.w_obs.dequantizer(m.weight_integer),
                    m.weight
                )

                print(f"____________rmse:{cs}_____________")

                mean_val     = m.normalize(x)
                mean_val_f   = m.float_foward_pass_normalize(x)
                dq_output    = m.betta_gamma_forward_pass(mean_val_f)
                dq_output_f  = m.float_betta_gamma_forward_pass(mean_val_f)

                diff = dq_output - dq_output_f

                flat_idx_max = diff.argmax()
                max_idx = torch.unravel_index(flat_idx_max, diff.shape)
                flat_idx_min = diff.argmin()
                min_idx = torch.unravel_index(flat_idx_min, diff.shape)

                if cs > 0.03:
                    search_scale_w = optimal_factor_search_lnorm(
                        m,
                        m.w_obs,
                        is_weight=False,
                        alpha=0.5,
                        betta=2.0,
                        itr=50
                    )
                    print("search_scale_w", search_scale_w)
                    update_scale_factor(
                        m,
                        m.w_obs,
                        m.w_obs.min_val,
                        m.w_obs.max_val,
                        search_scale_w
                    )

    model.unset_scale_opt()
    print()

import copy

# Deep copy for tokenizer (class instance)
tokenizer_copy = copy.deepcopy(tokenizer)

# Deep copy for sample (dict of strings & int)
sample_copy = copy.deepcopy(sample)

insert_proxi_opt(sample_copy, tokenizer_copy)

sampled_layers = {}
sample_layers(var_name="weight")
plot_distributions_icml()

# sampled_layers = {}

# sample_layers(var_name="else")
# plot_distributions_icml()

"""# Simple example - Quastion Answering Modeling"""

sentence1_pos = "The children are playing in the park."
sentence2_pos = "Kids are playing outside in the park."

sentence1_neg = "The children are playing in the park."
sentence2_neg = "The stock market closed lower today."

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_mrpc_example(sent1, sent2):
    inputs = tokenizer(
        sent1,
        sent2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits            # shape: [1, 2]

    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(logits, dim=-1).item()

    label_str = "paraphrase" if pred == 1 else "not paraphrase"

    return {
        "sentence1": sent1,
        "sentence2": sent2,
        "logits": logits.cpu().numpy(),
        "probs": probs.cpu().numpy(),
        "prediction": pred,
        "label": label_str,
    }


# Run positive (paraphrase)
res_pos = run_mrpc_example(sentence1_pos, sentence2_pos)

# Run negative (non-paraphrase)
res_neg = run_mrpc_example(sentence1_neg, sentence2_neg)

print("Sentence 1:", res_pos["sentence1"])
print("Sentence 2:", res_pos["sentence2"])
print("Prediction:", res_pos["label"])
print("Probabilities [not paraphrase, paraphrase]:", res_pos["probs"])
print()

print("Sentence 1:", res_neg["sentence1"])
print("Sentence 2:", res_neg["sentence2"])
print("Prediction:", res_neg["label"])
print("Probabilities [not paraphrase, paraphrase]:", res_neg["probs"])

"""## debug"""

def run_mean(layer, s_factor, verbose, metric, metric_name):
    mean_scale = layer.s1 * s_factor
    layer.scale_input_normed , layer.zero_point_input_normed = layer.in_obs_normalize.calculate_qparams()

    x_q = layer.in_obs_normalize.quantizer(layer.x_test)
    x_q = (x_q - layer.zero_point_input_normed).int()

    xq_sum = x_q.sum(dim=-1, keepdim=True)
    xq_scaled_mean = (xq_sum * mean_scale) / x_q.shape[-1]   # float division
    xq_mean = xq_scaled_mean.trunc().int()

    xdq_mean = xq_mean * layer.in_obs_normalize.scale / mean_scale
    x_mean = layer.x_test.float().mean(dim=-1, keepdim=True)

    local_metric_res = metric(
        xdq_mean,
        x_mean
    )

    if verbose:
        print("-----------------------------")
        print(f"[SCALE: {s_factor}]xdq_mean {xdq_mean-x_mean}")
        print(f"[SCALE: {s_factor}] {metric_name} { local_metric_res}")

    return local_metric_res, x_q, xq_mean


def run_ver(m, s_factor, x_q, xq_mean, verbose, metric, metric_name):
    var_scale = m.s2 * s_factor

    q_e_2_x = xq_mean * xq_mean
    q_e_x_2 = (((x_q**2).sum(dim=-1, keepdim=True) * var_scale) / x_q.shape[-1]).trunc().int()
    q_var_x = q_e_x_2 - q_e_2_x

    qdq_x_var = ((m.scale_input_normed) ** 2) * (q_var_x.float()) / var_scale
    x_var = m.x_test.var(dim=-1, unbiased=False, keepdim=True)

    local_metric_res = metric(
        qdq_x_var,
        x_var
    )

    if verbose:
        print("-----------------------------")
        # print(f"[SCALE: {s_factor}]xdq_mean {qdq_x_var-x_var}")
        print(f"[SCALE: {s_factor}] xdq_mean {qdq_x_var}")
        print(f"[SCALE: {s_factor}] x_var {x_var}")
        print(f"[SCALE: {s_factor}] {metric_name} { local_metric_res}")

    return local_metric_res

def mean_scale_lineseach(m, metric, metric_name, min_max="min", alpha=0.5, betta=2):
    global_metric_res, _, _ = run_mean(m, 1.0, False, metric, metric_name)
    opt_mean_scale = 1.0

    for s1_factor in np.linspace(alpha, betta, num=10):
        local_metric_res, _, _ = run_mean(m, s1_factor, False, metric, metric_name)
        print(f"[SCALE:{s1_factor}] {global_metric_res} vs. {local_metric_res}")

        is_update = global_metric_res > local_metric_res if min_max=="min" else global_metric_res < local_metric_res
        if is_update:
            global_metric_res = local_metric_res
            opt_mean_scale = s1_factor

    print(f"[OPT SCALE:{opt_mean_scale }] {global_metric_res} metric={metric_name}")
    local_metric_res, x_q, xq_mean = run_mean(m, opt_mean_scale, False, metric, metric_name)

    return local_metric_res, x_q, xq_mean, opt_mean_scale


def var_scale_linesearch(m, x_q, xq_mean, metric, metric_name, min_max="min", alpha=0.5, betta=2):
    global_metric_res = run_ver(m, 1.0, x_q, xq_mean, False, metric, metric_name)

    opt_var_scale = 1.0

    for s2_factor in np.linspace(alpha, betta, num=10):
        local_metric_res = run_ver(m,
            s2_factor,
            x_q, xq_mean,
            False,
            metric,
            metric_name
        )
        print(f"[SCALE:{s2_factor}] {global_metric_res} vs. {local_metric_res}")

        is_update = global_metric_res > local_metric_res if min_max=="min" else global_metric_res < local_metric_res
        if is_update:
            global_metric_res = local_metric_res
            opt_var_scale = s2_factor

    print(f"[OPT SCALE:{opt_var_scale }] {global_metric_res} metric={metric_name}")

    local_metric_res = run_ver(m, opt_var_scale, x_q, xq_mean, False, metric, metric_name)

    return opt_var_scale

# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # --- Setup ---
# insert_proxi_opt(sample_copy, tokenizer_copy)

# sampled_layers = {}

# sns.set(style="whitegrid", font_scale=1.4)
# sample_layers(var_name="out")
# plot_distributions_icml()

"""# Evaluation"""

# -------------------------------------------------
# Setup
# -------------------------------------------------
import torch
from datasets import load_dataset
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

evaluation_config = experiment_config.get("evaluation", {})
BATCH_SIZE = evaluation_config.get("batch_size", 16)
MAX_LEN = evaluation_config.get("max_len", 128)
NUM_VAL_EXAMPLES = evaluation_config.get("num_val_examples", 408)  # full MRPC validation set

# IMPORTANT: tokenizer from MRPC checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GLUE MRPC validation set (streaming)
dataset = load_dataset(
    "nyu-mll/glue",
    "mrpc",
    split="validation",
    streaming=True
).shuffle(seed=400)

streamer = dataset.iter(batch_size=BATCH_SIZE)

# MRPC metrics
metric_acc = evaluate.load("glue", "mrpc")
metric_f1  = evaluate.load("f1")

predictions = []
references = []
losses = []

# =================================================

def get_batch_predictions(batch, tokenizer):
    inputs = tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs["token_type_ids"].to(device)
    labels = torch.tensor(batch["label"], device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return preds, outputs.loss.detach().cpu().item() if outputs.loss is not None else None


# Iterate through batches
for i, batch in enumerate(tqdm(streamer, total=NUM_VAL_EXAMPLES // BATCH_SIZE)):

    if i * BATCH_SIZE >= NUM_VAL_EXAMPLES:
        break

    batch_preds, batch_loss = get_batch_predictions(batch, tokenizer)
    if batch_loss is not None:
        losses.append(batch_loss)

    for j in range(len(batch_preds)):
        predictions.append(batch_preds[j])
        references.append(batch["label"][j])

    # ✅ Interim metrics every 10 batches
    if (i + 1) % 10 == 0:
        acc_res = metric_acc.compute(
            predictions=predictions,
            references=references,
        )
        f1_res = metric_f1.compute(
            predictions=predictions,
            references=references,
        )
        print(
            f"\n[Batch {i+1}] "
            f"Accuracy: {acc_res['accuracy']:.4f}, "
            f"F1: {f1_res['f1']:.4f}\n"
        )

# Final metrics
acc_res = metric_acc.compute(
    predictions=predictions,
    references=references,
)
f1_res = metric_f1.compute(
    predictions=predictions,
    references=references,
)

print("Final Accuracy:", acc_res["accuracy"])
print("Final F1:", f1_res["f1"])

average_loss = sum(losses) / len(losses) if losses else None
print("Final Loss:", average_loss)

result_path = save_experiment_result(
    accuracy=acc_res["accuracy"],
    loss=average_loss,
    configuration=experiment_config,
    quantized=experiment_config.get("q_module_list", []),
    output_dir=PROJECT_DIR / "output",
    extra={
        "metrics": {
            "accuracy": acc_res["accuracy"],
            "f1": f1_res["f1"],
        },
        "primary_metric_name": "accuracy",
        "primary_metric_value": acc_res["accuracy"],
        "model_name": model_name,
        "num_val_examples": len(references),
    },
)
print("Saved results:", result_path)

# # -------------------------------------------------
# # Setup
# # -------------------------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hf_model.eval()
# hf_model.to(device)

# BATCH_SIZE = 16
# MAX_LEN = 128
# NUM_VAL_EXAMPLES = 1043  # full CoLA validation set

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# # GLUE CoLA validation set (streaming)
# dataset = load_dataset(
#     "glue",
#     "cola",
#     split="validation",
#     streaming=True
# ).shuffle(seed=400)

# streamer = dataset.iter(batch_size=BATCH_SIZE)

# # CoLA metric = Matthews Correlation Coefficient
# metric = evaluate.load("glue", "cola")

# predictions = []
# references = []

# # =================================================

# def get_batch_predictions(batch, tokenizer):
#     inputs = tokenizer(
#         batch["sentence"],
#         truncation=True,
#         padding="max_length",
#         max_length=MAX_LEN,
#         return_tensors="pt",
#     )

#     input_ids = inputs["input_ids"].to(device)
#     attention_mask = inputs["attention_mask"].to(device)

#     with torch.no_grad():
#         try:
#             outputs = hf_model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#             )

#         except RuntimeError as err:
#           print(err)
#           raise RuntimeError(err)

#     logits = outputs.logits
#     preds = torch.argmax(logits, dim=-1).cpu().tolist()
#     return preds


# # Iterate through batches
# for i, batch in enumerate(tqdm(streamer, total=NUM_VAL_EXAMPLES // BATCH_SIZE)):

#     if i * BATCH_SIZE >= NUM_VAL_EXAMPLES:
#         break

#     batch_preds = get_batch_predictions(batch, tokenizer)

#     for j in range(len(batch_preds)):
#         predictions.append(batch_preds[j])
#         references.append(batch["label"][j])

#     # ✅ Interim MCC every 10 batches
#     if (i + 1) % 10 == 0:
#         partial_results = metric.compute(
#             predictions=predictions,
#             references=references,
#         )
#         print(
#             f"\n[Batch {i+1}] Interim MCC: "
#             f"{partial_results['matthews_correlation']:.4f}\n"
#         )

# results = metric.compute(
#     predictions=predictions,
#     references=references,
# )

# print("Final MCC:", results["matthews_correlation"])

"""# LNORM STATS"""

# Models
#     # CoLA - geckos/bert-base-uncased-finetuned-glue-cola
#     # MRPC - lrs21/bert-base-uncased-finetuned-glue-mrpc
#     # SST-2 - raj-p/bert-base-uncased-finetuned-glue-sst2
#     # QQP - textattack/bert-base-uncased-QQP
#     # MNLI - jcbao77/bert-uncased-fine-tuned-zero-shot-baseline-mnli
#     # QNLI - mrm8488/bert-uncased-finetuned-qnli
#     # RTE - nickapch/bert-base-uncased-finetuned-glue_rte
#     # SQuAD - bert-large-uncased-whole-word-masking-finetuned-squad

# results
# baseline     - F1: 0.8981001727115717
# all          - F1: 0.8842832469775475
# norm no opt  - F1: 0.38746438746438744
