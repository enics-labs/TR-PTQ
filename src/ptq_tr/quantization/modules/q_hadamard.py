"""Quantized Hadamard product implementation."""

import torch
import torch.nn as nn

from ptq_tr.quantization.observers.minmax import MinMaxObserver


class qHadamardProd(nn.Module):
    """
    A quantized matrix multiplication layer using integer-only arithmetic for efficiency.
    This class uses MinMaxObserver to quantize inputs and perform matrix multiplication in the quantized domain.
    """

    def __init__(
        self,
        nof_bits1=16,
        nof_bits2=16,
        quant=False,
        is_calibrate=False,
    ):
        """
        Initializes the quantized matrix multiplication layer.

        Args:
            bias (bool): If set to False, the layer will not use bias. Default: True.
            is_calibrate (bool): Flag for enabling calibration mode to update observers.
        """
        super(qHadamardProd, self).__init__()
        self.nof_bits1 = nof_bits1
        self.nof_bits2 = nof_bits2
        self.nof_bits_o = self.nof_bits1 + self.nof_bits2

        self.in1_obs = MinMaxObserver(
            qscheme=torch.per_tensor_affine,
            nof_bits=self.nof_bits1,
            is_calibrate=is_calibrate,
            name="hadamard",
        )
        self.in2_obs = MinMaxObserver(
            qscheme=torch.per_tensor_affine,
            nof_bits=self.nof_bits2,
            is_calibrate=is_calibrate,
            name="hadamard",
        )
        self.out_obs = MinMaxObserver(
            qscheme=torch.per_tensor_symmetric,
            nof_bits=self.nof_bits_o,
            is_calibrate=is_calibrate,
            name="hadamard",
        )

        self.is_calibrate = is_calibrate
        self.quant = quant
        self.is_out_scaled = False

    def set_out_scale(self):
        """
        Quantizes weights and bias if they have not been quantized yet.
        Sets scale and zero-point for weights and bias, and stores quantized values.
        """
        if not self.is_out_scaled:
            self.out_obs.scale = self.in1_obs.scale * self.in2_obs.scale
            self.out_obs.zero_point = 0
            self.is_out_scaled = True

    def float_forward_pass(self, x1, x2):
        return x1 * x2

    def forward_pass(self, x1, x2):
        self.set_out_scale()

        _, zero_point_input1 = self.in1_obs.calculate_qparams()
        _, zero_point_input2 = self.in2_obs.calculate_qparams()

        x_q1 = self.in1_obs.quantizer(x1)
        x_q2 = self.in2_obs.quantizer(x2)

        output_q = (x_q1 - zero_point_input1) * (x_q2 - zero_point_input2)

        dq_output = self.out_obs.dequantizer(output_q)
        return dq_output

    def forward(self, x1, x2):
        """
        Forward pass for matrix multiplication.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: Result of matrix multiplication, either quantized or floating-point.
        """
        if self.is_calibrate:
            self.in1_obs(x1)
            self.in2_obs(x2)
            return self.float_forward_pass(x1, x2)

        if self.quant:
            return self.forward_pass(x1, x2)
        else:
            return self.float_forward_pass(x1, x2)

    def set_calibration_flag(self):
        """Enable calibration mode to update observers."""
        self.is_calibrate = True

    def unset_calibration_flag(self):
        """Disable calibration mode."""
        self.is_calibrate = False

    def set_quant(self):
        self.quant = True

    def unset_quant(self):
        self.quant = False

    def set_scale_opt(self):
        self.is_opt_scale = True

    def unset_scale_opt(self):
        self.is_opt_scale = False
