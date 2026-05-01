import torch
import torch.nn as nn

from mrcp_quant.observers import MinMaxObserver


class QuantizedMatmul(nn.Module):
    """
    A quantized matrix multiplication layer using integer-only arithmetic for efficiency.
    This class uses MinMaxObserver to quantize inputs and perform matrix multiplication in the quantized domain.
    """

    def __init__(self,
                 is_calibrate=False,
                 in1_bits=8,
                 in2_bits=8,
                 is_opt_scale=False,
                 quant=True):
        """
        Initializes the quantized matrix multiplication layer.

        Args:
            bias (bool): If set to False, the layer will not use bias. Default: True.
            is_calibrate (bool): Flag for enabling calibration mode to update observers.
        """
        super(QuantizedMatmul, self).__init__()

        self.in1_bits=in1_bits
        self.in2_bits=in2_bits
        self.is_calibrate = is_calibrate
        self.quant = quant  # Whether to use quantized operations
        self.is_weights_quantized = False
        self.is_opt_scale = is_opt_scale

        # Initialize observers for two input tensors and the output
        self.in1_obs = MinMaxObserver(qscheme=torch.per_tensor_affine,
                                      nof_bits=self.in1_bits,
                                      is_calibrate=is_calibrate,
                                      name="MatMul")
        self.in2_obs = MinMaxObserver(qscheme=torch.per_tensor_affine,
                                      nof_bits=self.in2_bits,
                                      is_calibrate=is_calibrate,
                                      name="MatMul")
        self.out_obs = MinMaxObserver(qscheme=torch.per_tensor_symmetric,
                                      nof_bits=(self.in1_bits+self.in2_bits),
                                      is_calibrate=is_calibrate,
                                      name="MatMul")

    def quantize_weights_and_bias(self):
        """
        Quantizes weights and bias if they have not been quantized yet.
        Sets scale and zero-point for weights and bias, and stores quantized values.
        """
        if not self.is_weights_quantized:

            # Calculate bias quantization scale as the product of input and weight scales
            self.out_obs.scale = self.in1_obs.scale * self.in2_obs.scale
            self.out_obs.zero_point = 0  # Bias zero-point is typically set to 0

            # Mark weights and bias as quantized
            self.is_weights_quantized = True


    def forward_pass(self, x1, x2):
        self.quantize_weights_and_bias()

        # Quantize the inputs
        scale_input1, zero_point_input1 = self.in1_obs.calculate_qparams()
        scale_input2, zero_point_input2 = self.in2_obs.calculate_qparams()

        # Quantize the inputs based on their scales and zero points
        x_q1 = self.in1_obs.quantizer(x1)
        x_q2 = self.in2_obs.quantizer(x2)

        # Perform quantized matrix multiplication
        output_q = (x_q1 - zero_point_input1) @ (x_q2 - zero_point_input2)

        # Dequantize the output
        dq_output = self.out_obs.dequantizer(output_q)
        return dq_output

    def float_forward_pass(self, x1, x2):
        return x1 @ x2

    def forward(self, x1, x2):
        """
        Forward pass for matrix multiplication.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: Result of matrix multiplication, either quantized or floating-point.
        """
        # Calibration mode: update observers with input min/max values
        if self.is_calibrate:
            self.in1_obs(x1)
            self.in2_obs(x2)
            return self.float_forward_pass(x1, x2)  # Return the floating-point result during calibration

        if self.is_opt_scale:
            self.opt_input1 = x1
            self.opt_input2 = x2
            return self.float_forward_pass(x1, x2)  # Return the floating-point result during calibration

        # Quantized matrix multiplication
        if self.quant:
            return self.forward_pass(x1, x2)
        else:
            # Standard floating-point matrix multiplication
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
        del self.opt_input1
        del self.opt_input2
        self.is_opt_scale = False

"""## Softmax"""

sf_layer_idx = 0
