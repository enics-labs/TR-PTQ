"""Min-max observer definitions extracted from the notebook."""

import torch

from ptq_tr.quantization.observers.base import ObserverBase


class MinMaxObserver(ObserverBase):
    def __init__(
        self,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        nof_bits=8,
        is_calibrate=False,
        name="base",
    ):
        """
        MinMaxObserver class is used to record the minimum and maximum values of the input activations.
        These values will be used to compute the scale and zero-point for quantization.

        Args:
            dtype (torch.dtype): The data type for quantization, e.g., torch.qint8.
            qscheme (torch.qscheme): The quantization scheme, e.g., torch.per_tensor_affine.
            nof_bits (int): Number of bits for quantization.
            is_calibrate (bool): Flag indicating whether to calculate the scale and zero-point during forwarding.
        """
        super(MinMaxObserver, self).__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.nof_bits = nof_bits
        self.is_calibrate = is_calibrate
        self.min_val = torch.tensor(float("inf"))
        self.max_val = torch.tensor(float("-inf"))
        self.zero_point = None
        self.scale = None
        self.name = name
        self.ez = None
        self.nml = None
        self.sz = None

    def forward(self, x):
        """
        This method updates the minimum and maximum values by comparing the current min and max
        with those in the input tensor. If is_calibrate is True, it calculates the scale and zero-point.

        Args:
            x (torch.Tensor): The input tensor to observe.
        """
        global_min = x.min()
        global_max = x.max()

        if global_min < self.min_val:
            self.min_val = global_min
        if global_max > self.max_val:
            self.max_val = global_max

        self.calculate_qparams()
        self.ez = x.element_size()
        self.nml = x.numel()
        self.sz = x.size()

        return x

    def calculate_qparams(self):
        """
        This method computes the scale and zero-point for quantization based on the min and max values
        observed during the forward pass.

        Returns:
            scale (torch.Tensor): The quantization scale.
            zero_point (torch.Tensor): The quantization zero-point.
        """
        if self.min_val == self.max_val:
            self.scale = torch.tensor([1.0], dtype=torch.float32)
            self.zero_point = torch.tensor([0], dtype=torch.int32)
        else:
            self.scale = (self.max_val - self.min_val) / (2**self.nof_bits - 1)
            if self.qscheme == torch.per_tensor_affine:
                self.zero_point = torch.round(-self.min_val / self.scale).clamp(0, 2**self.nof_bits - 1)
            else:
                self.zero_point = 0
        return self.scale, self.zero_point

    def quantizer(self, x, scale=None):
        """
        Quantizes the input tensor using the calculated scale and zero-point.

        Args:
            x (torch.Tensor): The input tensor to quantize.

        Returns:
            torch.Tensor: The quantized tensor.
        """
        if self.scale is None or self.zero_point is None:
            raise ValueError(f"Scale and zero-point must be calculated before quantizing [{self.name}].")

        if self.qscheme == torch.per_tensor_affine:
            x_q = (x / (self.scale) + self.zero_point).round().clamp(0, 2**self.nof_bits - 1)
        else:
            x_q = (
                (x / (self.scale) + self.zero_point)
                .round()
                .clamp(-(2 ** (self.nof_bits - 1)), 2 ** (self.nof_bits - 1) - 1)
            )

        return x_q.int()

    def dequantizer(self, x_q):
        """
        Dequantizes the input tensor using the calculated scale and zero-point.

        Args:
            x_q (torch.Tensor): The quantized tensor to dequantize.

        Returns:
            torch.Tensor: The dequantized tensor.
        """
        if self.scale is None or self.zero_point is None:
            raise ValueError(f"Scale and zero-point must be calculated before dequantizing [{self.name}].")

        x_fp = (self.scale) * (x_q.float() - self.zero_point)
        return x_fp
