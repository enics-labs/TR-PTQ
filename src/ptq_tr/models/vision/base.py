"""Shared base classes for vision transformer models."""

import torch
import torch.nn as nn

from ptq_tr.quantization.modules.q_hadamard import qHadamardProd
from ptq_tr.quantization.modules.q_layernorm import QLayerNorm
from ptq_tr.quantization.modules.quant_linear import QuantizedLinear
from ptq_tr.quantization.modules.quant_matmul import QuantizedMatmul
from ptq_tr.quantization.qparams import QauntParams


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (batch-wise)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class QuantTransformer(QauntParams):
    def __init__(
        self,
        quant=False,
        is_calibrate=False,
        q_module_list=[QuantizedLinear, QuantizedMatmul, qHadamardProd, QLayerNorm],
    ):
        super().__init__()
        self.q_module_list = q_module_list
        self.quant = quant
        self.is_calibrate = is_calibrate
        self.q_module_list = q_module_list

    def set_calibration_flag(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                print(f"[SET CALIBRATION FLAG] type:{type(m)}, module list: {self.q_module_list}")
                m.set_calibration_flag()

    def unset_calibration_flag(self):
        for m in self.modules():
            if type(m) in self.q_module_list:
                print(f"[UNSET CALIBRATION FLAG] type:{type(m)}, module list: {self.q_module_list}")
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
