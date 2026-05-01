# MRPC Quantized BERT Experiment

This project contains a split-out version of the original `copy_of_bert_glue_mrcp.py` experiment plus a matching Jupyter notebook. The quantized model classes live in `mrcp_quant/`, and experiment settings are controlled by `quant_config.json`.

## Files

| Path | Purpose |
| --- | --- |
| `copy_of_bert_glue_mrcp.py` | Python entry point for the MRPC quantization experiment. |
| `copy_of_bert_glue_mrcp.ipynb` | Notebook version of the same entry point. |
| `run_glue_quant.py` | Generic configurable GLUE runner for MRPC, CoLA, RTE, SST-2, QQP, MNLI, and QNLI. |
| `quant_config.json` | Main experiment config. Edit this to change quantized layers, bit widths, and run sizes. |
| `quant_config_cola.json` | Example CoLA config for the generic GLUE runner. |
| `quant_config_rte.json` | Example RTE config for the generic GLUE runner. |
| `quant_config_sst2.json` | Example SST-2 config for the generic GLUE runner. |
| `quant_config_qqp.json` | Example QQP config for the generic GLUE runner. |
| `quant_config_mnli.json` | Example MNLI config for the generic GLUE runner. |
| `quant_config_qnli.json` | Example QNLI config for the generic GLUE runner. |
| `mrcp_quant/observers.py` | Observer classes. |
| `mrcp_quant/quant_utils.py` | Integer approximation and lookup-table helpers. |
| `mrcp_quant/quantized_layers.py` | Compatibility exports for quantized layers. |
| `mrcp_quant/layers/` | Quantized layer implementations split by layer type. |
| `mrcp_quant/bert_model.py` | Custom BERT body and task-independent heads. |
| `mrcp_quant/heads.py` | Sequence classification and MRPC heads. |
| `mrcp_quant/config.py` | Config loading and quantized-module name resolution. |

## Using A Config

By default, the script and notebook read:

```text
quant_config.json
```

To use another config file from the Python script, set `MRPC_QUANT_CONFIG`:

```bash
MRPC_QUANT_CONFIG=my_config.json python copy_of_bert_glue_mrcp.py
```

In the notebook, edit the `CONFIG_PATH` cell or set the environment variable before running the notebook.

## Generic GLUE Runner

Use `run_glue_quant.py` for task-configurable MRPC, CoLA, RTE, SST-2, QQP, MNLI, or QNLI runs:

```bash
GLUE_QUANT_CONFIG=quant_config_cola.json python run_glue_quant.py
GLUE_QUANT_CONFIG=quant_config_rte.json python run_glue_quant.py
GLUE_QUANT_CONFIG=quant_config_sst2.json python run_glue_quant.py
GLUE_QUANT_CONFIG=quant_config_qqp.json python run_glue_quant.py
GLUE_QUANT_CONFIG=quant_config_mnli.json python run_glue_quant.py
GLUE_QUANT_CONFIG=quant_config_qnli.json python run_glue_quant.py
```

Set `task_name` to `mrpc`, `cola`, `rte`, `sst2`, `qqp`, `mnli`, or `qnli`. The runner chooses the matching custom head, tokenizer fields, GLUE split, metric, calibration flow, and timestamped JSON output.

## Top-Level Options

| Option | Type | Default in `quant_config.json` | Description |
| --- | --- | --- | --- |
| `model_name` | string | `lrs21/bert-base-uncased-finetuned-glue-mrpc` | Hugging Face checkpoint used for tokenizer, config, and reference weights. |
| `task_name` | string | `mrpc` | Task used by `run_glue_quant.py`; valid values are `mrpc`, `cola`, `rte`, `sst2`, `qqp`, `mnli`, and `qnli`. |
| `q_module_list` | list of strings | `["QLayerNorm"]` | Quantized layer classes to enable with `model.set_quant()`. |
| `quantization.defaults` | object | See table below | Default bit/LUT settings applied before model construction. |
| `quantization.layers` | object | `{}` | Optional per-module overrides keyed by `model.named_modules()` path. |
| `calibration` | object | `{"num_samples": 32}` | Calibration loop settings. |
| `scale_optimization` | object | `{"num_samples": 30}` | Scale optimization loop settings. |
| `plotting` | object | `{"sample_batches": 3, "max_samples": 12}` | Layer sampling and plot settings. |
| `evaluation` | object | `{"batch_size": 16, "max_len": 128, "num_val_examples": 408}` | Evaluation loop settings. |

## Valid `q_module_list` Values

| Name | Class | What It Enables |
| --- | --- | --- |
| `QLayerNorm` | `mrcp_quant.quantized_layers.QLayerNorm` | Quantized LayerNorm path. |
| `IntSoftmaxTS` | `mrcp_quant.quantized_layers.IntSoftmaxTS` | Integer/Taylor softmax path. |
| `QuantizedLinear` | `mrcp_quant.quantized_layers.QuantizedLinear` | Quantized linear layers. |
| `QuantizedMatmul` | `mrcp_quant.quantized_layers.QuantizedMatmul` | Quantized matrix multiplication. |
| `IntGeluTS` | `mrcp_quant.quantized_layers.IntGeluTS` | Integer/Taylor GELU path. |
| `qHadamardProd` | `mrcp_quant.quantized_layers.qHadamardProd` | Quantized Hadamard product helper. |

Example:

```json
{
  "q_module_list": [
    "QLayerNorm",
    "QuantizedLinear",
    "QuantizedMatmul"
  ]
}
```

## Quantization Parameter Options

These keys are valid inside `quantization.defaults` and each `quantization.layers.<module_path>` override. The older `quant_params`, `module_quant_params`, `model_quant_params`, and `layer_quant_params` keys are still accepted for compatibility.

| Key | Type | Default | Used By |
| --- | --- | --- | --- |
| `quant` | boolean | `false` | Initial quant flag before `set_quant()` is called. |
| `nof_bits_linear1` | integer | `8` | Input bit width for `QuantizedLinear`. |
| `nof_bits_linear2` | integer | `8` | Weight bit width for `QuantizedLinear`. |
| `nof_bits_gelu` | integer | `8` | Bit width for `IntGeluTS`. |
| `lut_size_gelu` | integer | `16` | LUT size for `IntGeluTS`. |
| `split_table_gelu` | integer | `-1` | Split-table setting passed to `TaylorExponent` inside `IntGeluTS`. |
| `nof_bits_softmax` | integer | `8` | Bit width for `IntSoftmaxTS`. |
| `lut_size_softmax` | integer | `7` | LUT size for `IntSoftmaxTS`. |
| `split_table_softmax` | integer | `0` | Split-table setting passed to `TaylorExponent` inside `IntSoftmaxTS`. |
| `nof_bits_lnorm1` | integer | `12` | First/input bit width for `QLayerNorm`. |
| `nof_bits_lnorm2` | integer | `12` | Second/weight bit width for `QLayerNorm`. |
| `split_table_lnorm` | integer | `-1` | Split-table setting passed to `TaylorExponent` inside `QLayerNorm`. |
| `nof_bits_matmul1` | integer | `8` | First input bit width for `QuantizedMatmul`. |
| `nof_bits_matmul2` | integer | `8` | Second input bit width for `QuantizedMatmul`. |

Example layer override:

```json
{
  "quantization": {
    "layers": {
      "bert.encoder.layer.0.intermediate.intermediate_act_fn": {
        "nof_bits_gelu": 10,
        "lut_size_gelu": 32
      }
    }
  }
}
```

## Calibration Options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `calibration.num_samples` | integer | `32` | Number of streaming MRPC train examples used to collect observer statistics. |

## Scale Optimization Options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `scale_optimization.num_samples` | integer | `30` | Number of streaming MRPC train examples used during scale-factor optimization. |

## Plotting Options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `plotting.sample_batches` | integer | `3` | Number of streamed samples to inspect while collecting plotted layer values. |
| `plotting.max_samples` | integer | `12` | Maximum number of `QLayerNorm` layers to collect for plotting. |

## Evaluation Options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `evaluation.batch_size` | integer | `16` | Validation batch size. |
| `evaluation.max_len` | integer | `128` | Tokenizer max sequence length. |
| `evaluation.num_val_examples` | integer | `408` | Number of MRPC validation examples to evaluate. |

## Minimal Example

```json
{
  "model_name": "lrs21/bert-base-uncased-finetuned-glue-mrpc",
  "q_module_list": ["QLayerNorm"],
  "quantization": {
    "defaults": {
      "nof_bits_lnorm1": 12,
      "nof_bits_lnorm2": 12
    },
    "layers": {
      "bert.encoder.layer.0.output.LayerNorm": {
        "nof_bits_lnorm2": 8
      }
    }
  },
  "evaluation": {
    "batch_size": 16,
    "max_len": 128,
    "num_val_examples": 408
  }
}
```

Missing config sections fall back to the defaults in the code.
