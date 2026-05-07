import json
from pathlib import Path

from .bert_model import DEFAULT_QUANT_PARAMS, set_default_quant_params
from .layers import (
    IntGeluTS,
    IntSoftmaxTS,
    QLayerNorm,
    QuantizedLinear,
    QuantizedMatmul,
    qHadamardProd,
)


Q_MODULES = {
    "QLayerNorm": QLayerNorm,
    "IntSoftmaxTS": IntSoftmaxTS,
    "QuantizedLinear": QuantizedLinear,
    "QuantizedMatmul": QuantizedMatmul,
    "IntGeluTS": IntGeluTS,
    "qHadamardProd": qHadamardProd,
}


def load_experiment_config(path="notebooks/configs/quant_config.json"):
    config_path = Path(path)
    if not config_path.exists() and not config_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        repo_config_path = repo_root / config_path
        notebooks_config_path = repo_root / "notebooks" / "configs" / config_path.name
        if repo_config_path.exists():
            config_path = repo_config_path
        elif notebooks_config_path.exists():
            config_path = notebooks_config_path
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_q_module_list(module_names):
    unknown = sorted(set(module_names) - set(Q_MODULES))
    if unknown:
        valid = ", ".join(sorted(Q_MODULES))
        raise ValueError(f"Unknown q_module_list entries {unknown}. Valid entries: {valid}")
    return [Q_MODULES[name] for name in module_names]


def _collect_quant_param_overrides(config):
    quantization = config.get("quantization", {})
    overrides = {}
    overrides.update(config.get("quant_params", {}))
    overrides.update(quantization.get("defaults", {}))
    return overrides


def apply_experiment_config(config):
    quant_params = _collect_quant_param_overrides(config)
    if quant_params:
        set_default_quant_params(quant_params, target="all")

    module_quant_params = config.get("module_quant_params", {})
    if module_quant_params:
        set_default_quant_params(module_quant_params, target="module")

    model_quant_params = config.get("model_quant_params", {})
    if model_quant_params:
        set_default_quant_params(model_quant_params, target="model")


def _validate_quant_params(params, context):
    unknown = sorted(set(params) - set(DEFAULT_QUANT_PARAMS))
    if unknown:
        valid = ", ".join(sorted(DEFAULT_QUANT_PARAMS))
        raise ValueError(f"Unknown quantization parameter(s) in {context}: {unknown}. Valid entries: {valid}")


def _set_observer_bits(observer, nof_bits):
    if observer is not None:
        observer.nof_bits = nof_bits
        observer.scale = None
        observer.zero_point = None


def _apply_quant_params_to_module(module, params):
    _validate_quant_params(params, type(module).__name__)

    for key, value in params.items():
        setattr(module, key, value)

    if "quant" in params and hasattr(module, "quant"):
        module.quant = params["quant"]

    is_linear = isinstance(module, QuantizedLinear)
    is_layernorm = isinstance(module, QLayerNorm)
    is_gelu = isinstance(module, IntGeluTS)
    is_softmax = isinstance(module, IntSoftmaxTS)
    is_matmul = isinstance(module, QuantizedMatmul)

    if is_linear and "nof_bits_linear1" in params:
        _set_observer_bits(getattr(module, "in_obs", None), params["nof_bits_linear1"])
    if is_linear and "nof_bits_linear2" in params:
        _set_observer_bits(getattr(module, "w_obs", None), params["nof_bits_linear2"])
    if is_linear and ("nof_bits_linear1" in params or "nof_bits_linear2" in params):
        bits1 = getattr(module, "nof_bits1", params.get("nof_bits_linear1"))
        bits2 = getattr(module, "nof_bits2", params.get("nof_bits_linear2"))
        if bits1 is not None and bits2 is not None:
            module.nof_bits1 = params.get("nof_bits_linear1", bits1)
            module.nof_bits2 = params.get("nof_bits_linear2", bits2)
            module.nof_bits_b = module.nof_bits1 + module.nof_bits2
            _set_observer_bits(getattr(module, "b_obs", None), module.nof_bits_b)

    if is_layernorm and "nof_bits_lnorm1" in params:
        for attr in ("in_obs", "in_obs_normalize"):
            _set_observer_bits(getattr(module, attr, None), params["nof_bits_lnorm1"])
        if hasattr(module, "in1_bits"):
            module.in1_bits = params["nof_bits_lnorm1"]
    if is_layernorm and "nof_bits_lnorm2" in params:
        _set_observer_bits(getattr(module, "w_obs", None), params["nof_bits_lnorm2"])
        if hasattr(module, "in2_bits"):
            module.in2_bits = params["nof_bits_lnorm2"]
    if is_layernorm and "split_table_lnorm" in params:
        module.split_table = params["split_table_lnorm"]

    if is_gelu and "nof_bits_gelu" in params:
        module.nof_bits = params["nof_bits_gelu"]
        module.input_bits = module.nof_bits - 4
        module.output_bits = module.nof_bits + 1
        _set_observer_bits(getattr(module, "in_obs", None), module.nof_bits)
    if is_gelu and "lut_size_gelu" in params:
        module.LUT_SIZE = params["lut_size_gelu"]
    if is_gelu and "split_table_gelu" in params:
        module.split_table = params["split_table_gelu"]

    if is_softmax and "nof_bits_softmax" in params:
        module.nof_bits = params["nof_bits_softmax"]
        module.input_bits = module.nof_bits - 4
        module.output_bits = module.nof_bits + 1
        _set_observer_bits(getattr(module, "in_obs", None), module.nof_bits)
    if is_softmax and "lut_size_softmax" in params:
        module.LUT_SIZE = params["lut_size_softmax"]
    if is_softmax and "split_table_softmax" in params:
        module.split_table = params["split_table_softmax"]

    if is_matmul and "nof_bits_matmul1" in params:
        _set_observer_bits(getattr(module, "in1_obs", None), params["nof_bits_matmul1"])
        if hasattr(module, "in1_bits"):
            module.in1_bits = params["nof_bits_matmul1"]
    if is_matmul and "nof_bits_matmul2" in params:
        _set_observer_bits(getattr(module, "in2_obs", None), params["nof_bits_matmul2"])
        if hasattr(module, "in2_bits"):
            module.in2_bits = params["nof_bits_matmul2"]

    if hasattr(module, "is_weights_quantized"):
        module.is_weights_quantized = False


def _collect_layer_quant_overrides(config):
    quantization = config.get("quantization", {})
    return {
        **config.get("layer_quant_params", {}),
        **quantization.get("layers", {}),
    }


def apply_layer_quant_overrides(model, config):
    layer_overrides = _collect_layer_quant_overrides(config)
    if not layer_overrides:
        return {}

    modules = dict(model.named_modules())
    applied = {}

    for module_path, params in layer_overrides.items():
        if module_path not in modules:
            raise ValueError(f"Unknown module path in layer quantization overrides: {module_path}")
        _apply_quant_params_to_module(modules[module_path], params)
        applied[module_path] = params

    return applied
