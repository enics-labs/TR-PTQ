"""Base observer definitions extracted from the notebook."""

import torch
import torch.nn as nn


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
