"""Similarity metrics."""

import torch


def cosine_similarity(tensor1, tensor2):
    """Calculates cosine similarity between two tensors."""
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    dot_product = torch.dot(tensor1, tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    return dot_product / (norm1 * norm2)


def nrmse(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8, norm: str = "var") -> torch.Tensor:
    """
    Compute Normalized Root Mean Squared Error (NRMSE).

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor (same shape as y_true).
        eps: Small constant to avoid division by zero.
        norm: Normalization method:
              - "var": normalize by sqrt(Var(y_true))  (default, scale-invariant)
              - "minmax": normalize by (max(y_true) - min(y_true))
              - "mean": normalize by mean(|y_true|)
              - "none": no normalization (just RMSE)

    Returns:
        Scalar tensor with NRMSE value.
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)

    if norm == "var":
        denom = torch.sqrt(y_true.var(unbiased=False) + eps)
    elif norm == "minmax":
        denom = (y_true.max() - y_true.min()).clamp(min=eps)
    elif norm == "mean":
        denom = y_true.abs().mean().clamp(min=eps)
    elif norm == "none":
        denom = 1.0
    else:
        raise ValueError(f"Unknown normalization method: {norm}")

    return rmse / denom
