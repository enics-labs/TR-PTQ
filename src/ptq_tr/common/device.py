"""Device selection helpers."""

import torch


def get_default_device():
    if torch.cuda.is_available():
        print("GPU is available.")
        return torch.device("cuda")
    print("GPU is not available, using CPU.")
    return torch.device("cpu")
