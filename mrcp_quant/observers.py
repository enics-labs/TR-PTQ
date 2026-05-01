import torch
import torch.nn as nn
import numpy as np

class ObserverBase(nn.Module):
    def __init__(self, dtype=torch.qint8, qscheme=torch.per_tensor_affine):
        """
        ObserverBase class is a base class for observers in PyTorch quantization.

        Args:
            dtype (torch.dtype): Data type for quantization, typically `torch.qint8` or `torch.quint8`.
            qscheme (torch.qscheme): Quantization scheme. For example, `torch.per_tensor_affine`.
        """
        super(ObserverBase, self).__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.is_symmetric = qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]

    def forward(self, x):
        """
        Placeholder for the forward method that will process the input data (activations or weights).
        To be overridden in derived classes.

        Args:
            x (torch.Tensor): The input tensor to observe.
        """
        raise NotImplementedError("ObserverBase.forward must be implemented in derived classes")

    def calculate_qparams(self):
        """
        Placeholder for the calculate_qparams method. It should calculate the quantization parameters
        (scale and zero point) based on the statistics collected in the forward pass.
        """
        raise NotImplementedError("ObserverBase.calculate_qparams must be implemented in derived classes")

class MinMaxObserver(ObserverBase):
    def __init__(self,
                 dtype=torch.qint8,
                 qscheme=torch.per_tensor_affine,
                 nof_bits=8,
                 is_calibrate=False,
                 name='base'):
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
        self.min_val = torch.tensor(float('inf'))  # Initialize min with infinity
        self.max_val = torch.tensor(float('-inf'))  # Initialize max with -infinity
        self.zero_point = None
        self.scale = None
        self.name = name
        self.ez = None
        self.nml = None
        self.sz  = None

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

        # Calculate scale and zero-point during forwarding if is_calibrate is True
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
            # To handle cases where min == max to avoid division by zero
            self.scale = torch.tensor([1.0], dtype=torch.float32)
            self.zero_point = torch.tensor([0], dtype=torch.int32)
        else:
            # Calculate scale and zero-point for qint8 quantization
            self.scale = (self.max_val - self.min_val) / (2 ** self.nof_bits - 1)
            if self.qscheme == torch.per_tensor_affine:
                self.zero_point = torch.round(-self.min_val / self.scale).clamp(0, 2 ** self.nof_bits - 1)
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
        # if self.is_calibrate:
        if self.scale is None or self.zero_point is None:
            raise ValueError(f"Scale and zero-point must be calculated before quantizing [{self.name}].")

        # Quantize the input
        if self.qscheme == torch.per_tensor_affine:
            x_q = (x / (self.scale) + self.zero_point).round().clamp(0, 2 ** self.nof_bits - 1)
        else:
            x_q = (x / (self.scale) + self.zero_point).round().clamp(-2 ** (self.nof_bits-1), 2 ** (self.nof_bits-1) - 1)

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

        # Dequantize the input
        x_fp = (self.scale) * (x_q.float() - self.zero_point)
        # x_fp = (self.scale) * (x_q.float())
        return x_fp

import torch

class PerChannelMinMaxObserver(ObserverBase):
    def __init__(self,
                 ch_axis=-1,  # Dimension along which to apply per-channel quantization
                 dtype=torch.qint8,
                 qscheme=torch.per_channel_affine,
                 nof_bits=8,
                 name='per_channel'):
        """
        PerChannelMinMaxObserver tracks min/max values for each channel separately.

        Args:
            ch_axis (int): The channel axis (e.g., 0 for weights [Out_ch, In_ch]).
        """
        super(PerChannelMinMaxObserver, self).__init__()
        self.ch_axis = ch_axis
        self.dtype = dtype
        self.qscheme = qscheme
        self.nof_bits = nof_bits
        self.name = name

        # Initialize as None; they will be shaped during the first forward pass
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.zero_point = None

    def forward(self, x):
        # Determine the shape for reduction: all dims except ch_axis
        # Example: if x is [C, H, W] and ch_axis=0, we reduce over (1, 2)
        dims = list(range(x.dim()))
        dims.pop(self.ch_axis)

        # Calculate per-channel min/max
        # .values is used because torch.min returns (values, indices)
        ch_min = x.amin(dim=dims)
        ch_max = x.amax(dim=dims)

        print(x.shape)
        if self.min_val is None:
            self.min_val = ch_min
            self.max_val = ch_max
        else:
            self.min_val = torch.min(self.min_val, ch_min)
            self.max_val = torch.max(self.max_val, ch_max)

        self.calculate_qparams()
        return x

    def calculate_qparams(self):
        if self.min_val is None or self.max_val is None:
            return

        # Handle min == max to avoid division by zero per channel
        diff = self.max_val - self.min_val
        diff[diff == 0] = 1.0

        # Per-channel scales
        self.scale = diff / (2 ** self.nof_bits - 1)

        if self.qscheme == torch.per_channel_affine:
            self.zero_point = torch.round(-self.min_val / self.scale).clamp(0, 2 ** self.nof_bits - 1)
        else:
            # Symmetric quantization
            self.zero_point = torch.zeros_like(self.scale)

        return self.scale, self.zero_point

    def quantizer(self, x):
        if self.scale is None:
            raise ValueError(f"Observer {self.name} has not been calibrated.")

        # Reshape scale and zero_point to be broadcastable with x
        shape = [1] * x.dim()
        shape[self.ch_axis] = -1
        s = self.scale.reshape(shape)
        zp = self.zero_point.reshape(shape)

        if self.qscheme == torch.per_channel_affine:
            x_q = (x / s + zp).round().clamp(0, 2 ** self.nof_bits - 1)
        else:
            q_min = -2 ** (self.nof_bits - 1)
            q_max = 2 ** (self.nof_bits - 1) - 1
            x_q = (x / s + zp).round().clamp(q_min, q_max)

        return x_q.int()

    def dequantizer(self, x_q):
        shape = [1] * x_q.dim()
        shape[self.ch_axis] = -1
        s = self.scale.reshape(shape)
        zp = self.zero_point.reshape(shape)

        return s * (x_q.float() - zp)

class PercentileMinMaxObserver(MinMaxObserver):
    def __init__(self,
                 dtype=torch.qint8,
                 qscheme=torch.per_tensor_affine,
                 nof_bits=8,
                 is_calibrate=False,
                 name='percentile_minmax',
                 percentile=0.1,
                 symmetric=False):
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            nof_bits=nof_bits,
            is_calibrate=is_calibrate,
            name=name,
        )

        self.percentile = percentile
        self.symmetric = symmetric

        # buffer for collecting values
        self._values = []

    def forward(self, x):
        """
        Collect values for percentile-based min/max.
        """
        x_detached = x.detach().flatten()
        self._values.append(x_detached)
        return x

    def calculate_qparams(self):
        """
        Override only the statistics computation,
        then reuse MinMaxObserver.calculate_qparams().
        """
        if len(self._values) == 0:
            raise RuntimeError(f"[{self.name}] No data collected for percentile observer")

        values = torch.cat(self._values)

        lo = torch.quantile(values, self.percentile / 100.0)
        hi = torch.quantile(values, 1.0 - self.percentile / 100.0)

        if self.symmetric:
            max_abs = torch.max(lo.abs(), hi.abs())
            self.min_val = -max_abs
            self.max_val =  max_abs
        else:
            self.min_val = lo
            self.max_val = hi

        # Now reuse the original MinMax logic
        return super().calculate_qparams()

import torch
import torch.nn as nn
import numpy as np


class HistogramObserver(ObserverBase):
    def __init__(self,
                 dtype=torch.qint8,
                 qscheme=torch.per_tensor_symmetric,
                 nof_bits=8,
                 bins=2048,
                 is_calibrate=False,
                 name="hist"):
        super().__init__(dtype=dtype, qscheme=qscheme)

        self.is_calibrate = is_calibrate
        self.nof_bits = nof_bits
        self.bins = bins
        self.name = name

        self.hist = None
        self.min_val = None
        self.max_val = None

        self.scale = None
        self.zero_point = None

    # ------------------------------------------------------------
    # Forward: collect histogram
    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        x = x.detach()

        x_min = x.min()
        x_max = x.max()

        if self.min_val is None:
            self.min_val = x_min
            self.max_val = x_max
        else:
            self.min_val = torch.minimum(self.min_val, x_min)
            self.max_val = torch.maximum(self.max_val, x_max)

        # Histogram range is fixed after first pass
        if self.hist is None:
            self.hist = torch.histc(
                x,
                bins=self.bins,
                min=self.min_val.item(),
                max=self.max_val.item()
            )
        else:
            self.hist += torch.histc(
                x,
                bins=self.bins,
                min=self.min_val.item(),
                max=self.max_val.item()
            )

        return x


    # ------------------------------------------------------------
    # KL-divergence helper
    # ------------------------------------------------------------
    @staticmethod
    def _kl_divergence(p, q, eps=1e-8):
        p = p + eps
        q = q + eps
        return torch.sum(p * torch.log(p / q))

    # ------------------------------------------------------------
    # Find optimal clipping threshold using KL
    # ------------------------------------------------------------
    def _find_optimal_threshold(self):
        hist = self.hist.float()
        hist_sum = hist.sum()
        pdf = hist / hist_sum

        num_quant_bins = 2 ** self.nof_bits - 1
        best_kl = float("inf")
        best_bin = None

        for clip_bin in range(num_quant_bins, self.bins):
            # Reference distribution P
            p = pdf[:clip_bin].clone()
            p[-1] += pdf[clip_bin:].sum()

            # Quantized distribution Q
            q = torch.zeros_like(p)

            bin_width = clip_bin / num_quant_bins

            for i in range(num_quant_bins):
                start = int(i * bin_width)
                end = int((i + 1) * bin_width)
                end = min(end, clip_bin)

                if start >= end:
                    continue

                q[start:end] = p[start:end].sum() / (end - start)

            kl = self._kl_divergence(p, q)

            if kl < best_kl:
                best_kl = kl
                best_bin = clip_bin

        return best_bin

    def get_qparams(self):
        return self.scale, self.zero_point

    # ------------------------------------------------------------
    # Calculate scale and zero-point
    # ------------------------------------------------------------
    def calculate_qparams(self):
        # ------------------------------------------------------------
        # SAFETY CHECK: KL requires histogram bins >= quant bins
        # ------------------------------------------------------------
        qbins = 2 ** self.nof_bits - 1
        if self.bins < qbins:
            raise ValueError(
                f"[HistogramObserver:{self.name}] "
                f"Invalid configuration: bins={self.bins} < "
                f"quantization bins={qbins}. "
                "KL-divergence calibration is impossible. "
                "Use bins >= 2^bits - 1 (e.g., 512 or 2048)."
            )

        if self.hist is None:
            raise RuntimeError(
                f"[HistogramObserver:{self.name}] "
                "No histogram collected. Run calibration first."
            )

        # ------------------------------------------------------------
        # Existing KL logic continues below
        # ------------------------------------------------------------
        optimal_bin = self._find_optimal_threshold()

        bin_width = (self.max_val - self.min_val) / self.bins
        threshold = self.min_val + bin_width * optimal_bin

        max_abs = torch.max(threshold.abs())
        qmax = 2 ** (self.nof_bits - 1) - 1

        self.scale = max_abs / qmax
        self.zero_point = torch.tensor(0, dtype=torch.int32)

        return self.scale, self.zero_point


    def quantizer(self, x, scale=None):
        """
        Quantizes the input tensor using the calculated scale and zero-point.

        Args:
            x (torch.Tensor): The input tensor to quantize.

        Returns:
            torch.Tensor: The quantized tensor.
        """
        # if self.is_calibrate:
        if self.scale is None or self.zero_point is None:
            raise ValueError(f"Scale and zero-point must be calculated before quantizing [{self.name}].")

        # Quantize the input
        if self.qscheme == torch.per_tensor_affine:
            x_q = (x / (self.scale) + self.zero_point).round().clamp(0, 2 ** self.nof_bits - 1)
        else:
            x_q = (x / (self.scale) + self.zero_point).round().clamp(-2 ** (self.nof_bits-1), 2 ** (self.nof_bits-1) - 1)

        return x_q.int()

    # ------------------------------------------------------------
    # Dequantizer
    # ------------------------------------------------------------
    def dequantizer(self, x_q):
        if self.scale is None:
            raise RuntimeError(f"[{self.name}] scale not initialized")

        return x_q.float() * self.scale
