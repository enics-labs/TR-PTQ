import torch


def cosine_similarity(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculate cosine similarity between two tensors."""
    reference = reference.flatten()
    candidate = candidate.flatten()
    return torch.dot(reference, candidate) / (torch.norm(reference) * torch.norm(candidate) + eps)


def nrmse(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    eps: float = 1e-8,
    norm: str = "var",
) -> torch.Tensor:
    """Compute normalized root mean squared error."""
    rmse = torch.sqrt(torch.mean((reference - candidate) ** 2))

    if norm == "var":
        denominator = torch.sqrt(reference.var(unbiased=False) + eps)
    elif norm == "minmax":
        denominator = (reference.max() - reference.min()).clamp(min=eps)
    elif norm == "mean":
        denominator = reference.abs().mean().clamp(min=eps)
    elif norm == "none":
        denominator = 1.0
    else:
        raise ValueError(f"Unknown normalization method: {norm}")

    return rmse / denominator
