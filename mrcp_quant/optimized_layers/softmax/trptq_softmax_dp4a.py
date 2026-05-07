"""Python shim for the CUDA extension.

This ensures PyTorch shared libraries are loaded before importing the compiled
extension module, avoiding libc10/libtorch loader errors in notebook workflows.
"""

import torch  # noqa: F401 - intentionally imported for side effects
from _trptq_softmax_dp4a import *  # noqa: F401,F403


A_MIN = -8
A_MAX = 0
EXP_SCALE_U8 = 255


def build_exp_lut_u8(device="cuda"):
    anchors = torch.arange(A_MIN, A_MAX + 1, dtype=torch.float32, device=device)
    lut = torch.exp(anchors)
    return torch.round(lut / lut.max() * EXP_SCALE_U8).to(torch.uint8)


def _default_lut(exp_lut, device):
    if exp_lut is not None:
        return exp_lut.contiguous()
    return build_exp_lut_u8(device=device)


def _check_recip_input(x):
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.int32:
        raise ValueError("x must be int32")


def reciprocal_u8(x):
    _check_recip_input(x)

    x = x.contiguous()
    out = torch.empty_like(x, dtype=torch.uint8)
    recip_u8(x, out)
    return out


def reciprocal_u16(x):
    _check_recip_input(x)

    x = x.contiguous()
    out = torch.empty_like(x, dtype=torch.uint16)
    recip_u16(x, out)
    return out


def reciprocal_q16_i32(x):
    _check_recip_input(x)

    x = x.contiguous()
    out = torch.empty_like(x, dtype=torch.int32)
    recip_q16_i32(x, out)
    return out


def _check_sigmoid_input(x_q44, bits):
    if not x_q44.is_cuda:
        raise ValueError("x_q44 must be a CUDA tensor")
    if x_q44.dtype != torch.int8:
        raise ValueError("x_q44 must be int8 Q4.4")
    if bits < 4:
        raise ValueError("bits must be >= 4 so ln output can be converted to Q4.4")


def sigmoid_u8_from_q44(x_q44, exp_lut=None, bits=8):
    """Fused sigmoid approximation using CUDA TR-exp and CUDA TR-ln.

    Args:
        x_q44: int8 CUDA tensor in Q4.4 format.
        exp_lut: optional uint8 CUDA LUT with exp anchors [-8, ..., 0].
        bits: scale bits passed to ln_arith. With the uint8 exp kernel, use 8.

    Returns:
        uint8 CUDA tensor with sigmoid(x) scaled by 255.
    """
    _check_sigmoid_input(x_q44, bits)

    x_q44 = x_q44.contiguous()
    lut = _default_lut(exp_lut, x_q44.device)
    out = torch.empty_like(x_q44, dtype=torch.uint8)
    sigmoid_arith(x_q44, lut, bits, out)
    return out


def staged_sigmoid_u8_from_q44(x_q44, exp_lut=None, bits=8):
    """Debug version of sigmoid_u8_from_q44 using separate exp/ln kernels."""
    _check_sigmoid_input(x_q44, bits)

    x_q44 = x_q44.contiguous()
    lut = _default_lut(exp_lut, x_q44.device)
    x_neg = torch.clamp(x_q44, max=0).contiguous()
    x_pos_neg = (-torch.clamp(x_q44, min=0)).clamp(-128, 0).to(torch.int8).contiguous()

    exp_int = torch.empty_like(x_q44, dtype=torch.uint8)
    exp_zero = torch.empty_like(x_q44, dtype=torch.uint8)
    exp_arith_dp4a(x_neg, lut, exp_int)
    exp_arith_dp4a(x_pos_neg, lut, exp_zero)

    exp_sum = exp_int.to(torch.int32) + exp_zero.to(torch.int32)
    ln_sum = torch.empty_like(exp_sum)
    ln_arith(exp_sum.contiguous(), bits, ln_sum)

    neg_ln_q44 = (-(ln_sum >> (bits - 4))).clamp(-128, 0).to(torch.int8).contiguous()
    ln_mul = torch.empty_like(x_q44, dtype=torch.uint8)
    exp_arith_dp4a(neg_ln_q44, lut, ln_mul)

    sigmoid = (ln_mul.to(torch.int32) * exp_int.to(torch.int32) + (EXP_SCALE_U8 // 2))
    sigmoid = (sigmoid // EXP_SCALE_U8).clamp(0, EXP_SCALE_U8)
    return sigmoid.to(torch.uint8)


def gelu_sigmoid_u8_from_q44(x_q44, exp_lut=None, bits=8):
    return sigmoid_u8_from_q44(x_q44, exp_lut=exp_lut, bits=bits)


def gelu_sigmoid_dequant_from_q44(x_q44, exp_lut=None, bits=8):
    return gelu_sigmoid_u8_from_q44(x_q44, exp_lut=exp_lut, bits=bits).float() / EXP_SCALE_U8


def gelu_sigmoid_dequant(x, alpha=1.702, exp_lut=None, bits=8):
    x_q44 = torch.round(x * alpha * 16.0).clamp(-128, 127).to(torch.int8)
    return gelu_sigmoid_dequant_from_q44(x_q44, exp_lut=exp_lut, bits=bits)


def gelu_dequant(x, alpha=1.702, exp_lut=None, bits=8):
    return x * gelu_sigmoid_dequant(x, alpha=alpha, exp_lut=exp_lut, bits=bits)
