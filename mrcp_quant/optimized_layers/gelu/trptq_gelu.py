"""Python shim for the TR-PTQ GELU CUDA extension."""

import torch  # noqa: F401 - intentionally imported for PyTorch shared libs
from _trptq_gelu import *  # noqa: F401,F403


TR_A_MIN = -8
TR_A_MAX = 0
EXP_SCALE_U8 = 255
DEFAULT_ALPHA_TABLE = (
    2.834596,
    2.338217,
    1.978175,
    1.754823,
    1.642511,
    1.642511,
    1.754823,
    1.978175,
    2.338217,
    2.834596,
)


def build_exp_lut_u8(device="cuda"):
    anchors = torch.arange(TR_A_MIN, TR_A_MAX + 1, dtype=torch.float32, device=device)
    lut = torch.exp(anchors)
    return torch.round(lut / lut.max() * EXP_SCALE_U8).to(torch.uint8)


def _default_lut(exp_lut, device):
    if exp_lut is not None:
        return exp_lut.contiguous()
    return build_exp_lut_u8(device=device)


def _default_alpha_lut(alpha_lut, device):
    if alpha_lut is not None:
        if not torch.is_tensor(alpha_lut):
            alpha_lut = torch.tensor(alpha_lut)
        if alpha_lut.numel() != 10:
            raise ValueError("alpha_lut must contain exactly 10 values")
        return alpha_lut.to(device=device, dtype=torch.float32).contiguous()
    return torch.tensor(DEFAULT_ALPHA_TABLE, device=device, dtype=torch.float32)


def sigmoid_u8_from_q44(x_q44, exp_lut=None, bits=8):
    if not x_q44.is_cuda:
        raise ValueError("x_q44 must be a CUDA tensor")
    if x_q44.dtype != torch.int8:
        raise ValueError("x_q44 must be int8 Q4.4")

    x_q44 = x_q44.contiguous()
    lut = _default_lut(exp_lut, x_q44.device)
    out = torch.empty_like(x_q44, dtype=torch.uint8)
    sigmoid_arith(x_q44, lut, bits, out)
    return out


def sigmoid_dequant_from_q44(x_q44, exp_lut=None, bits=8):
    return sigmoid_u8_from_q44(x_q44, exp_lut=exp_lut, bits=bits).float() / EXP_SCALE_U8


def sigmoid_mixed_u8_from_q44(x_q44, exp_lut=None, bits=8):
    if not x_q44.is_cuda:
        raise ValueError("x_q44 must be a CUDA tensor")
    if x_q44.dtype != torch.int8:
        raise ValueError("x_q44 must be int8 Q4.4")

    x_q44 = x_q44.contiguous()
    lut = _default_lut(exp_lut, x_q44.device)
    out = torch.empty_like(x_q44, dtype=torch.uint8)
    if "sigmoid_mixed_arith" not in globals():
        raise RuntimeError(
            "The loaded _trptq_gelu extension does not export sigmoid_mixed_arith. "
            "Reinstall/rebuild the GELU extension, then restart the notebook kernel."
        )
    sigmoid_mixed_arith(x_q44, lut, bits, out)
    return out


def sigmoid_mixed_dequant_from_q44(x_q44, exp_lut=None, bits=8):
    return sigmoid_mixed_u8_from_q44(x_q44, exp_lut=exp_lut, bits=bits).float() / EXP_SCALE_U8


def sigmoid_recip_u8_from_q44(x_q44, exp_lut=None):
    if not x_q44.is_cuda:
        raise ValueError("x_q44 must be a CUDA tensor")
    if x_q44.dtype != torch.int8:
        raise ValueError("x_q44 must be int8 Q4.4")

    x_q44 = x_q44.contiguous()
    lut = _default_lut(exp_lut, x_q44.device)
    out = torch.empty_like(x_q44, dtype=torch.uint8)
    if "sigmoid_recip_arith" not in globals():
        raise RuntimeError(
            "The loaded _trptq_gelu extension does not export sigmoid_recip_arith. "
            "Reinstall/rebuild the GELU extension, then restart the notebook kernel."
        )
    sigmoid_recip_arith(x_q44, lut, out)
    return out


def sigmoid_recip_dequant_from_q44(x_q44, exp_lut=None):
    return sigmoid_recip_u8_from_q44(x_q44, exp_lut=exp_lut).float() / EXP_SCALE_U8


def sigmoid_alpha_u8(x, alpha=1.702, exp_lut=None, bits=8):
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")

    x = x.contiguous()
    lut = _default_lut(exp_lut, x.device)
    out = torch.empty_like(x, dtype=torch.uint8)
    if "sigmoid_alpha_arith" not in globals():
        raise RuntimeError(
            "The loaded _trptq_gelu extension does not export sigmoid_alpha_arith. "
            "Reinstall/rebuild the GELU extension, then restart the notebook kernel."
        )
    sigmoid_alpha_arith(x, lut, float(alpha), bits, out)
    return out


def sigmoid_alpha_dequant(x, alpha=1.702, exp_lut=None, bits=8):
    return sigmoid_alpha_u8(x, alpha=alpha, exp_lut=exp_lut, bits=bits).float() / EXP_SCALE_U8


def sigmoid_alpha_mixed_u8(x, alpha=1.702, exp_lut=None, bits=8):
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")

    x = x.contiguous()
    lut = _default_lut(exp_lut, x.device)
    out = torch.empty_like(x, dtype=torch.uint8)
    if "sigmoid_alpha_mixed_arith" not in globals():
        raise RuntimeError(
            "The loaded _trptq_gelu extension does not export sigmoid_alpha_mixed_arith. "
            "Reinstall/rebuild the GELU extension, then restart the notebook kernel."
        )
    sigmoid_alpha_mixed_arith(x, lut, float(alpha), bits, out)
    return out


def sigmoid_alpha_mixed_dequant(x, alpha=1.702, exp_lut=None, bits=8):
    return sigmoid_alpha_mixed_u8(x, alpha=alpha, exp_lut=exp_lut, bits=bits).float() / EXP_SCALE_U8


def exp_variant_u8_from_q44(x_q44, variant, exp_lut=None):
    if not x_q44.is_cuda:
        raise ValueError("x_q44 must be a CUDA tensor")
    if x_q44.dtype != torch.int8:
        raise ValueError("x_q44 must be int8 Q4.4")

    variant_ids = {
        "0rd": 0,
        "1rd": 1,
        "linear": 1,
        "2rd": 2,
    }
    if isinstance(variant, str):
        try:
            variant = variant_ids[variant]
        except KeyError as exc:
            raise ValueError("variant must be 0, 1, 2, '0rd', '1rd', 'linear', or '2rd'") from exc
    if variant not in (0, 1, 2):
        raise ValueError("variant must be 0, 1, or 2")

    x_q44 = x_q44.contiguous()
    lut = _default_lut(exp_lut, x_q44.device)
    out = torch.empty_like(x_q44, dtype=torch.uint8)
    if "exp_variant_arith" not in globals():
        raise RuntimeError(
            "The loaded _trptq_gelu extension does not export exp_variant_arith. "
            "Reinstall/rebuild the GELU extension, then restart the notebook kernel."
        )
    exp_variant_arith(x_q44, lut, int(variant), out)
    return out


def gelu(x, alpha=1.702, exp_lut=None, bits=8):
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")

    x = x.contiguous()
    lut = _default_lut(exp_lut, x.device)
    out = torch.empty_like(x)
    gelu_arith(x, lut, float(alpha), bits, out)
    return out


def gelu_alpha_table(x, alpha_lut=None, exp_lut=None, bits=8):
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")

    x = x.contiguous()
    lut = _default_lut(exp_lut, x.device)
    alpha = _default_alpha_lut(alpha_lut, x.device)
    out = torch.empty_like(x)
    if "gelu_alpha_table_arith" not in globals():
        raise RuntimeError(
            "The loaded _trptq_gelu extension does not export gelu_alpha_table_arith. "
            "Reinstall/rebuild the GELU extension, then restart the notebook kernel."
        )
    gelu_alpha_table_arith(x, lut, alpha, bits, out)
    return out
