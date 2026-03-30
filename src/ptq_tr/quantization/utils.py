"""Shared quantization helpers."""

import torch


list_dist_exp = []


def safe_shift_l(x, shift):
    shift_left = torch.clamp(shift, min=0)
    shift_right = torch.clamp(-shift, min=0)
    return (x << shift_left) >> shift_right


def safe_shift_r(x, shift):
    shift_left = torch.clamp(shift, min=0)
    shift_right = torch.clamp(-shift, min=0)
    return (x >> shift_left) << shift_right


def build_exp_lut(nof_bits=16, LUT_SIZE=16):
    scale = 1 << (nof_bits - 1)
    exp_lut = torch.zeros(LUT_SIZE + LUT_SIZE, dtype=torch.int64)
    for i, offset in enumerate(range(-LUT_SIZE, LUT_SIZE)):
        exp_lut[i] = torch.round(torch.exp(torch.tensor(offset).float()) * scale).to(dtype=torch.int32)

    return exp_lut


def build_ln_lut(nof_bits=16, LUT_SIZE=16, eps=1e-5):
    scale = 1 << (nof_bits - 1)
    ln_lut = torch.zeros(LUT_SIZE + 1, dtype=torch.int64)
    ln_lut[0] = torch.round(torch.log(torch.tensor(0.25 + eps).float()) * scale).to(dtype=torch.int32)
    for i, offset in enumerate(range(1, LUT_SIZE)):
        ln_lut[i] = torch.round(torch.log(torch.tensor(2 ** (offset - 1) + eps).float()) * scale).to(
            dtype=torch.int32
        )

    return ln_lut


def int_approx_lower_power_of_two(x: torch.Tensor):
    lut_idx = torch.where(
        x == 0,
        torch.tensor(0, dtype=torch.int32, device=x.device),
        x.log2().floor().to(torch.int32),
    )
    return lut_idx


def TylorNLog(in_x, ev_point=-1, nof_bits=16, iterations=3, LUT_SIZE=16, ln_lut=None, LN2=None):
    """
    Approximate natural logarithm (ln) using a Taylor series expansion around evaluation points.

    Args:
        in_x (torch.Tensor): Input tensor.
        ev_point (int): Evaluation point for the Taylor series expansion. Default is -1 (automatically determined).
        nof_bits (int): Number of bits for quantization.
        iterations (int): Number of iterations for the Taylor series expansion.
        LUT_SIZE (int): Size of the lookup table (LUT).

    Returns:
        torch.Tensor: Approximated logarithmic value using Taylor series.
    """
    dtype = torch.int32
    dtype_w = dtype = torch.int32

    scale = torch.tensor(1 << (nof_bits - 1)).to(dtype)

    lut_idx = (int_approx_lower_power_of_two(in_x >> (nof_bits - 4)) - 3).clamp_(-4, LUT_SIZE + 4)

    k = (lut_idx << (nof_bits - 1)).to(dtype)
    sum_result = (k >> 1) + (k >> 3) + (k >> 4)

    if iterations == 0:
        return sum_result

    x_diff = (in_x - (1 << (nof_bits + lut_idx - 1))).to(dtype_w)
    div_scale = (1 << (15 - lut_idx)).to(dtype)
    x_pow = ((x_diff * div_scale) >> (15)).to(dtype)

    sum_result += x_pow

    if iterations == 1:
        return sum_result

    x_pow2 = ((x_pow * x_pow) >> (nof_bits)).to(dtype)
    sum_result -= x_pow2

    if iterations == 2:
        return sum_result

    x_pow3 = ((x_pow2 * x_pow) // (scale * 3)).to(dtype)
    sum_result += x_pow3

    return sum_result


def new_ln(xq, bits):
    aq = xq.log2().floor().int() - bits
    k1 = safe_shift_r(xq, aq)
    k2 = (aq - 1) << bits
    k = k1 + k2
    yq = (k >> 1) + (k >> 3) + (k >> 4)
    return yq


def TaylorExponent(x_scale, scale, iterations=1, input_bits=16, output_bits=16, LUT_SIZE=16, exp_lut=None):
    dtype = torch.int32
    udtype = torch.int32

    offset = ((x_scale + (1 << (input_bits - 2))) >> (input_bits - 1)).to(dtype)
    offset_clamped = offset.clamp_(-LUT_SIZE, LUT_SIZE)

    exp_offset = (offset_clamped.exp() * (1 << (output_bits - 1))).round().to(dtype=dtype)
    if iterations == 0:
        return exp_offset

    x_a = x_scale - (offset << (input_bits - 1))
    x_pow = x_a

    sum_result = x_pow + scale

    if iterations == 1:
        return (sum_result * exp_offset).to(udtype) >> (input_bits - 1)

    x_pow2 = (x_pow * x_pow) >> input_bits
    sum_result = sum_result + x_pow2

    if iterations == 2:
        return (sum_result * exp_offset).to(udtype) >> (input_bits - 1)

    x_pow3 = (x_pow2 * x_pow) // ((1 << (input_bits - 1)) * 3)
    sum_result = sum_result + x_pow3

    return (sum_result * exp_offset).to(udtype) >> (input_bits - 1)
