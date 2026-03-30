"""Quantization parameter containers shared by models."""

import torch.nn as nn


class QauntParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = False
        self.nof_bits_linear1 = 8
        self.nof_bits_linear2 = 8
        self.nof_bits_gelu = 8
        self.nof_bits_softmax = 8
        self.lut_size_softmax = 7
        self.nof_bits_lnorm1 = 12
        self.nof_bits_lnorm2 = 4
        self.nof_bits_matmul1 = 8
        self.nof_bits_matmul2 = 8

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
