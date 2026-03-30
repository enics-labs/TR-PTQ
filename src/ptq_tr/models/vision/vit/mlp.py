"""MLP block implementation."""

import torch.nn as nn

from ptq_tr.quantization.modules.int_gelu import IntGeluTS
from ptq_tr.quantization.modules.quant_linear import QuantizedLinear
from ptq_tr.quantization.qparams import QauntParams


class Mlp(QauntParams):
    """Feed-forward network (MLP) block used in Transformer blocks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=IntGeluTS,
        is_calibrate=False,
        quant=None,
        drop=0.0,
        quant_config=None,
    ):
        super().__init__(quant_config=quant_config)
        quant = self.quant if quant is None else quant
        self.quant = quant
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = QuantizedLinear(
            in_features,
            hidden_features,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
            quant=quant,
        )

        if act_layer is IntGeluTS:
            self.act = act_layer(
                quant=quant,
                LUT_SIZE=self.lut_size_gelu,
                nof_bits=self.nof_bits_gelu,
            )
        else:
            self.act = act_layer()

        self.fc2 = QuantizedLinear(
            hidden_features,
            out_features,
            nof_bits1=self.nof_bits_linear1,
            nof_bits2=self.nof_bits_linear2,
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
