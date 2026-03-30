"""Quantized linear layer implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptq_tr.quantization.observers.minmax import MinMaxObserver


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        nof_bits1=8,
        nof_bits2=8,
        qscheme=torch.per_tensor_affine,
        is_calibrate=False,
        is_opt_scale=False,
        quant=False,
    ):
        """
        A quantized version of nn.Linear that uses MinMaxObserver to quantize weights, bias, and inputs.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If set to False, the layer will not learn an additive bias. Default: True.
            dtype (torch.dtype): Data type for quantization, typically qint8 or quint8.
            qscheme (torch.qscheme): Quantization scheme, typically per_tensor_affine.
            nof_bits (int): Number of bits for quantization.
            is_calibrate (bool): If True, the observer will update min/max values but not quantize the input.
        """
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)

        self.nof_bits1 = nof_bits1
        self.nof_bits2 = nof_bits2
        self.nof_bits_b = nof_bits1 + nof_bits2

        self.in_obs = MinMaxObserver(qscheme=torch.per_tensor_affine, nof_bits=nof_bits1, is_calibrate=True, name="linear")
        self.w_obs = MinMaxObserver(qscheme=torch.per_tensor_symmetric, nof_bits=nof_bits2, is_calibrate=True, name="linear")
        self.b_obs = MinMaxObserver(
            qscheme=torch.per_tensor_symmetric, nof_bits=self.nof_bits_b, is_calibrate=True, name="linear"
        )

        self.is_calibrate = is_calibrate
        self.is_opt_scale = is_opt_scale
        self.is_weights_quantized = False
        self.quant = quant

    def quantize_weights_and_bias(self):
        """
        Quantizes weights and bias if they have not been quantized yet.
        Sets scale and zero-point for weights and bias, and stores quantized values.
        """
        if not self.is_weights_quantized:
            self.w_obs(self.weight)
            self.scale_weight, self.zero_point_weight = self.w_obs.calculate_qparams()
            self.weight_integer = self.w_obs.quantizer(self.weight) - self.zero_point_weight

            if self.bias is not None:
                self.b_obs.scale = self.in_obs.scale * self.w_obs.scale
                self.b_obs.zero_point = 0
                self.bias_integer = self.b_obs.quantizer(self.bias)
            else:
                self.b_obs.scale = self.in_obs.scale * self.w_obs.scale
                self.b_obs.zero_point = 0
                self.bias_integer = None

            self.is_weights_quantized = True

    def forward_pass(self, x):
        """
        Forward pass with quantization:
        - Quantizes weights and bias if not already done.
        - Quantizes the input.
        - Performs linear transformation with quantized weights and bias.
        - Dequantizes the output before returning.
        """
        self.quantize_weights_and_bias()

        self.scale_input, self.zero_point_input = self.in_obs.calculate_qparams()
        _x_q = self.in_obs.quantizer(x)
        x_q = _x_q - self.zero_point_input.int()

        if self.bias is not None:
            output_q = F.linear(
                x_q.float(),
                weight=self.weight_integer.float(),
                bias=self.bias_integer.float(),
            )
        else:
            output_q = F.linear(
                x_q.float(),
                weight=self.weight_integer.float(),
            )

        dq_output = self.b_obs.dequantizer(output_q)
        return dq_output

    def float_forward_pass(self, x):
        return F.linear(
            x,
            self.weight,
            self.bias,
        )

    def forward(self, x):
        """
        Forward pass through the quantized linear layer.
        If is_calibrate is True, update the min/max values of the input observer without performing quantization.
        Otherwise, quantize the input, weights, and bias, perform integer-based linear operation, and dequantize the result.

        Args:
            x (torch.Tensor): Input tensor to the linear layer.

        Returns:
            torch.Tensor: Dequantized output after quantized linear transformation.
        """
        if self.is_calibrate:
            self.in_obs(x)
            return self.float_forward_pass(x)

        if self.is_opt_scale:
            self.opt_input = x
            return self.float_forward_pass(x)

        if self.quant:
            return self.forward_pass(x)
        return self.float_forward_pass(x)

    def set_calibration_flag(self):
        self.is_calibrate = True

    def unset_calibration_flag(self):
        self.is_calibrate = False

    def set_quant(self):
        self.quant = True

    def unset_quant(self):
        self.quant = False

    def set_scale_opt(self):
        self.is_opt_scale = True

    def unset_scale_opt(self):
        del self.opt_input
        self.is_opt_scale = False
