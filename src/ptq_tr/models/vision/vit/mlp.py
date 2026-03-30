"""MLP block implementation."""

import torch.nn as nn

from ptq_tr.quantization.modules.int_gelu import IntGeluTS
from ptq_tr.quantization.modules.quant_linear import QuantizedLinear


class Mlp(nn.Module):
    """Feed-forward network (MLP) block used in Transformer blocks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=IntGeluTS,
        is_calibrate=False,
        quant=False,
        drop=0.0,
    ):
        super().__init__()
        self.quant = quant
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = QuantizedLinear(
            in_features,
            hidden_features,
            nof_bits1=8,
            nof_bits2=8,
            quant=quant,
        )

        self.act = act_layer(
            quant=quant,
            LUT_SIZE=16,
            nof_bits=8,
        )

        self.fc2 = QuantizedLinear(
            hidden_features,
            out_features,
            nof_bits1=8,
            nof_bits2=8,
            quant=quant,
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def set_quant(self):
        self.quant = True

    def unset_quant(self):
        self.quant = False
