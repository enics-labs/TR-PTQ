"""Quantization parameter containers and helpers shared by models."""

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict

import torch.nn as nn


QUANT_PARAM_FIELDS = (
    "quant",
    "nof_bits_linear1",
    "nof_bits_linear2",
    "nof_bits_gelu",
    "lut_size_gelu",
    "nof_bits_softmax",
    "int_bits_softmax",
    "lut_size_softmax",
    "nof_bits_lnorm1",
    "nof_bits_lnorm2",
    "nof_bits_matmul1",
    "nof_bits_matmul2",
)


@dataclass
class QuantConfig:
    quant: bool = False
    nof_bits_linear1: int = 8
    nof_bits_linear2: int = 8
    nof_bits_gelu: int = 8
    lut_size_gelu: int = 16
    nof_bits_softmax: int = 8
    int_bits_softmax: int = 4
    lut_size_softmax: int = 7
    nof_bits_lnorm1: int = 12
    nof_bits_lnorm2: int = 4
    nof_bits_matmul1: int = 8
    nof_bits_matmul2: int = 8


def quant_config_to_dict(quant_config: Any) -> Dict[str, Any]:
    """Normalize supported quant config sources into a plain dict."""
    if quant_config is None:
        return {}

    if is_dataclass(quant_config):
        data = asdict(quant_config)
    elif isinstance(quant_config, dict):
        data = dict(quant_config)
    else:
        data = {
            field_name: getattr(quant_config, field_name)
            for field_name in QUANT_PARAM_FIELDS
            if hasattr(quant_config, field_name)
        }

    return {
        field_name: value
        for field_name, value in data.items()
        if field_name in QUANT_PARAM_FIELDS and value is not None
    }


def merge_quant_config(quant_config: Any = None, **overrides: Any) -> Dict[str, Any]:
    """Merge an external config with explicit overrides."""
    merged = quant_config_to_dict(quant_config)
    merged.update({key: value for key, value in overrides.items() if value is not None})
    return merged


def apply_quant_config(target: Any, quant_config: Any = None) -> Any:
    """Apply quantization attributes to a target object in-place."""
    for field_name, value in quant_config_to_dict(quant_config).items():
        setattr(target, field_name, value)
    return target


class QauntParams(nn.Module):
    def __init__(self, quant_config=None):
        super().__init__()
        apply_quant_config(self, QuantConfig())
        apply_quant_config(self, quant_config)

    def set_calibration_flag(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_calibration_flag()

    def unset_calibration_flag(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_calibration_flag()

    def set_quant(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_quant()

    def unset_quant(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_quant()

    def set_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.set_scale_opt()

    def unset_scale_opt(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                m.unset_scale_opt()
