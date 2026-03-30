"""Per-channel min-max observer definitions."""

import torch

from ptq_tr.quantization.observers.base import ObserverBase


class PerChannelMinMaxObserver(ObserverBase):
    def __init__(
        self,
        dtype=torch.qint8,
        qscheme=torch.per_channel_affine,
        nof_bits=8,
        ch_axis=0,
        is_calibrate=False,
        name="base",
    ):
        """
        Per-channel variant of the min-max observer.

        Args:
            dtype (torch.dtype): The data type for quantization, e.g., torch.qint8.
            qscheme (torch.qscheme): The quantization scheme, e.g., torch.per_channel_affine.
            nof_bits (int): Number of bits for quantization.
            ch_axis (int): Channel axis used to compute independent min/max statistics.
            is_calibrate (bool): Flag indicating whether this observer is used during calibration.
            name (str): A short observer name for error messages.
        """
        super(PerChannelMinMaxObserver, self).__init__(dtype=dtype, qscheme=qscheme)
        if qscheme not in (torch.per_channel_affine, torch.per_channel_symmetric):
            raise ValueError(
                f"PerChannelMinMaxObserver supports only per-channel qschemes, got: {qscheme}"
            )

        self.dtype = dtype
        self.qscheme = qscheme
        self.nof_bits = nof_bits
        self.ch_axis = ch_axis
        self.is_calibrate = is_calibrate
        self.min_val = None
        self.max_val = None
        self.zero_point = None
        self.scale = None
        self.name = name
        self.ez = None
        self.nml = None
        self.sz = None

    def _normalized_axis(self, ndim):
        axis = self.ch_axis
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise IndexError(f"Channel axis {self.ch_axis} is out of bounds for tensor with {ndim} dims.")
        return axis

    def _reduce_dims(self, x):
        axis = self._normalized_axis(x.ndim)
        return tuple(dim for dim in range(x.ndim) if dim != axis)

    def _reshape_qparam(self, x, qparam):
        shape = [1] * x.ndim
        shape[self._normalized_axis(x.ndim)] = x.shape[self._normalized_axis(x.ndim)]
        return qparam.view(shape)

    def forward(self, x):
        """
        Update per-channel min/max statistics from the current input tensor.

        Args:
            x (torch.Tensor): The input tensor to observe.
        """
        reduce_dims = self._reduce_dims(x)
        if len(reduce_dims) == 0:
            current_min = x
            current_max = x
        else:
            current_min = x.amin(dim=reduce_dims)
            current_max = x.amax(dim=reduce_dims)

        if self.min_val is None:
            self.min_val = current_min.detach()
            self.max_val = current_max.detach()
        else:
            self.min_val = torch.minimum(self.min_val.to(x.device), current_min.detach())
            self.max_val = torch.maximum(self.max_val.to(x.device), current_max.detach())

        self.calculate_qparams()
        self.ez = x.element_size()
        self.nml = x.numel()
        self.sz = x.size()
        return x

    def calculate_qparams(self):
        """
        Compute per-channel scale and zero-point from the observed min/max values.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Scale and zero-point tensors, one value per channel.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError(f"Min/max statistics must be collected before calculating qparams [{self.name}].")

        same_range = self.min_val == self.max_val
        raw_scale = (self.max_val - self.min_val) / (2**self.nof_bits - 1)
        ones = torch.ones_like(raw_scale, dtype=torch.float32)
        self.scale = torch.where(same_range, ones, raw_scale.to(torch.float32))

        if self.qscheme == torch.per_channel_affine:
            raw_zero_point = torch.round(-self.min_val / self.scale).clamp(0, 2**self.nof_bits - 1)
            self.zero_point = torch.where(
                same_range,
                torch.zeros_like(raw_zero_point, dtype=torch.float32),
                raw_zero_point,
            ).to(torch.int32)
        else:
            self.zero_point = torch.zeros_like(self.scale, dtype=torch.int32)

        return self.scale, self.zero_point

    def quantizer(self, x, scale=None):
        """
        Quantize the input tensor with per-channel scale and zero-point.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            scale: Unused, kept for compatibility with the per-tensor observer API.

        Returns:
            torch.Tensor: Quantized integer tensor.
        """
        if self.scale is None or self.zero_point is None:
            raise ValueError(f"Scale and zero-point must be calculated before quantizing [{self.name}].")

        scale_bc = self._reshape_qparam(x, self.scale.to(x.device))
        zero_point_bc = self._reshape_qparam(x, self.zero_point.to(x.device))

        if self.qscheme == torch.per_channel_affine:
            x_q = (x / scale_bc + zero_point_bc).round().clamp(0, 2**self.nof_bits - 1)
        else:
            x_q = (
                (x / scale_bc + zero_point_bc)
                .round()
                .clamp(-(2 ** (self.nof_bits - 1)), 2 ** (self.nof_bits - 1) - 1)
            )

        return x_q.int()

    def dequantizer(self, x_q):
        """
        Dequantize a per-channel quantized tensor.

        Args:
            x_q (torch.Tensor): Quantized tensor.

        Returns:
            torch.Tensor: Dequantized floating-point tensor.
        """
        if self.scale is None or self.zero_point is None:
            raise ValueError(f"Scale and zero-point must be calculated before dequantizing [{self.name}].")

        scale_bc = self._reshape_qparam(x_q, self.scale.to(x_q.device))
        zero_point_bc = self._reshape_qparam(x_q, self.zero_point.to(x_q.device))
        return scale_bc * (x_q.float() - zero_point_bc)
