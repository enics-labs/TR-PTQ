import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import trptq_gelu
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    import trptq_gelu

from mrcp_quant.quant_utils import new_ln


TR_A_MIN = -8
TR_A_MAX = 0
Q8_U8_A_MIN = -8
DEFAULT_ALPHA_TABLE = torch.tensor(
    [
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
    ],
    dtype=torch.float32,
)


def exp_ref_q44(x_q44: torch.Tensor, lut_u8: torch.Tensor):
    x_i32 = x_q44.to(torch.int32)
    a = torch.clamp((x_i32 + 8) >> 4, TR_A_MIN, TR_A_MAX)
    r = x_i32 - (a << 4)
    m = 16 + r
    e = (lut_u8[(a - TR_A_MIN).long()].to(torch.int32) * m) >> 4
    return e.clamp(0, 255).to(torch.uint8)


def sigmoid_ref_from_q44(x_q44: torch.Tensor, lut_u8: torch.Tensor):
    x_neg = torch.clamp(x_q44, max=0)
    x_pos_neg = (-torch.clamp(x_q44, min=0)).clamp(-128, 0).to(torch.int8)

    exp_int = exp_ref_q44(x_neg, lut_u8)
    exp_zero = exp_ref_q44(x_pos_neg, lut_u8)
    exp_sum = exp_int.to(torch.int32) + exp_zero.to(torch.int32)

    ln_sum = new_ln(exp_sum, 8)
    neg_ln_q44 = (-(ln_sum >> 4)).clamp(-128, 0).to(torch.int8)
    ln_mul = exp_ref_q44(neg_ln_q44, lut_u8)

    sigmoid = ln_mul.to(torch.int32) * exp_int.to(torch.int32)
    return (sigmoid >> 8).clamp(0, 255).to(torch.uint8)


def exp_ref_q44_variant(x_q44: torch.Tensor, lut_u8: torch.Tensor, variant: int):
    if variant == 0:
        return exp_ref_q44(x_q44, lut_u8)
    if variant == 1:
        return exp_ref_q44(x_q44, lut_u8)
    x_i32 = x_q44.to(torch.int32)
    a = torch.clamp((x_i32 + 8) >> 4, TR_A_MIN, TR_A_MAX)
    r = x_i32 - (a << 4)
    m = 16 + r + ((r * r) >> 5)
    e = (lut_u8[(a - TR_A_MIN).long()].to(torch.int32) * m) >> 4
    return e.clamp(0, 255).to(torch.uint8)


def sigmoid_mixed_ref_from_q44(x_q44: torch.Tensor, lut_u8: torch.Tensor):
    x_neg = torch.clamp(x_q44, max=0)
    x_pos_neg = (-torch.clamp(x_q44, min=0)).clamp(-128, 0).to(torch.int8)

    exp_int = exp_ref_q44_variant(x_neg, lut_u8, 0)
    exp_zero = exp_ref_q44_variant(x_pos_neg, lut_u8, 0)
    exp_sum = exp_int.to(torch.int32) + exp_zero.to(torch.int32)

    ln_sum = new_ln(exp_sum, 8)
    neg_ln_q44 = (-(ln_sum >> 4)).clamp(-128, 0).to(torch.int8)
    ln_mul = exp_ref_q44_variant(neg_ln_q44, lut_u8, 2)

    return ((ln_mul.to(torch.int32) * exp_int.to(torch.int32)) >> 8).clamp(0, 255).to(torch.uint8)


def exp_q8_u8_ref(z_q8: torch.Tensor):
    lut = torch.tensor([0, 0, 1, 2, 5, 13, 35, 94, 255], device=z_q8.device, dtype=torch.int32)
    z_q8 = z_q8.to(torch.int32)
    a = torch.clamp((z_q8 + 128) >> 8, Q8_U8_A_MIN, 0)
    r_q8 = z_q8 - (a << 8)
    m_q8 = 256 + r_q8
    y = (lut[(a - Q8_U8_A_MIN).long()] * m_q8) >> 8
    return y.clamp(0, 255).to(torch.uint8)


def new_ln_raw_i32_q8_ref(x: torch.Tensor):
    x = x.to(torch.int64)
    out = torch.zeros_like(x, dtype=torch.int64)
    pos = x > 0
    if pos.any():
        xp = x[pos]
        a = torch.floor(torch.log2(xp.double())).to(torch.int64)
        shift = a - 8
        k1 = torch.where(shift >= 0, xp >> shift, xp << (-shift))
        k2 = (a - 1) << 8
        k = k1 + k2
        out[pos] = (k >> 1) + (k >> 3) + (k >> 4)
    return out.to(torch.int32)


def reciprocal_u8_ref(x: torch.Tensor):
    x = x.to(torch.int32)
    ln_x_q8 = new_ln_raw_i32_q8_ref(x)
    y = exp_q8_u8_ref(-ln_x_q8).to(torch.int32)
    return torch.where(x > 0, y, torch.zeros_like(x, dtype=torch.int32))


def sigmoid_recip_ref_from_q44(x_q44: torch.Tensor, lut_u8: torch.Tensor):
    x_i32 = x_q44.to(torch.int32)
    neg_abs = -x_i32.abs().clamp(max=128)
    exp_abs = exp_ref_q44(neg_abs.to(torch.int8), lut_u8).to(torch.int32)
    denom = 255 + exp_abs
    recip = reciprocal_u8_ref(denom)
    numer = torch.where(x_i32 >= 0, torch.full_like(exp_abs, 255), exp_abs)
    return ((numer * recip + 128) >> 8).clamp(0, 255).to(torch.uint8)


def sigmoid_alpha_ref(x: torch.Tensor, lut_u8: torch.Tensor, alpha=1.702):
    x_q44 = torch.round(x * alpha * 16.0).clamp(-128, 127).to(torch.int8)
    return sigmoid_ref_from_q44(x_q44, lut_u8)


def sigmoid_alpha_mixed_ref(x: torch.Tensor, lut_u8: torch.Tensor, alpha=1.702):
    x_q44 = torch.round(x * alpha * 16.0).clamp(-128, 127).to(torch.int8)
    return sigmoid_mixed_ref_from_q44(x_q44, lut_u8)


def gelu_ref(x: torch.Tensor, lut_u8: torch.Tensor, alpha=1.702):
    x_q44 = torch.round(x * alpha * 16.0).clamp(-128, 127).to(torch.int8)
    x_neg = torch.clamp(x_q44, max=0)
    x_pos_neg = (-torch.clamp(x_q44, min=0)).clamp(-128, 0).to(torch.int8)

    exp_int = exp_ref_q44(x_neg, lut_u8)
    exp_zero = exp_ref_q44(x_pos_neg, lut_u8)
    exp_sum = exp_int.to(torch.int32) + exp_zero.to(torch.int32)

    ln_sum = new_ln(exp_sum, 8)
    neg_ln_q44 = (-(ln_sum >> 4)).clamp(-128, 0).to(torch.int8)
    ln_mul = exp_ref_q44(neg_ln_q44, lut_u8)

    sig = (ln_mul.to(torch.int32) * exp_int.to(torch.int32)).float() / 65536.0
    return x * sig


def alpha_table_for_x(x: torch.Tensor, alpha_lut: torch.Tensor):
    alpha_lut = alpha_lut.to(device=x.device, dtype=torch.float32)
    indices = torch.clamp((torch.clamp(x, -5, 5) + 5).floor().long(), 0, 9)
    return alpha_lut[indices]


def gelu_alpha_table_ref(x: torch.Tensor, lut_u8: torch.Tensor, alpha_lut: torch.Tensor):
    alpha = alpha_table_for_x(x, alpha_lut)
    x_q44 = torch.round(x * alpha * 16.0).clamp(-128, 127).to(torch.int8)
    x_neg = torch.clamp(x_q44, max=0)
    x_pos_neg = (-torch.clamp(x_q44, min=0)).clamp(-128, 0).to(torch.int8)

    exp_int = exp_ref_q44(x_neg, lut_u8)
    exp_zero = exp_ref_q44(x_pos_neg, lut_u8)
    exp_sum = exp_int.to(torch.int32) + exp_zero.to(torch.int32)

    ln_sum = new_ln(exp_sum, 8)
    neg_ln_q44 = (-(ln_sum >> 4)).clamp(-128, 0).to(torch.int8)
    ln_mul = exp_ref_q44(neg_ln_q44, lut_u8)

    sig = (ln_mul.to(torch.int32) * exp_int.to(torch.int32)).float() / 65536.0
    return x * sig


def main():
    device = "cuda"
    torch.manual_seed(0)
    lut = trptq_gelu.build_exp_lut_u8(device=device)

    x_q44 = torch.randint(-80, 81, (4, 16), device=device, dtype=torch.int8)
    sigmoid_out = trptq_gelu.sigmoid_u8_from_q44(x_q44, lut)
    sigmoid_ref = sigmoid_ref_from_q44(x_q44, lut)
    torch.testing.assert_close(sigmoid_out, sigmoid_ref, rtol=0, atol=0)

    sigmoid_mixed_out = trptq_gelu.sigmoid_mixed_u8_from_q44(x_q44, lut)
    sigmoid_mixed_ref = sigmoid_mixed_ref_from_q44(x_q44, lut)
    torch.testing.assert_close(sigmoid_mixed_out, sigmoid_mixed_ref, rtol=0, atol=0)

    sigmoid_recip_out = trptq_gelu.sigmoid_recip_u8_from_q44(x_q44, lut)
    sigmoid_recip_ref = sigmoid_recip_ref_from_q44(x_q44, lut)
    torch.testing.assert_close(sigmoid_recip_out, sigmoid_recip_ref, rtol=0, atol=0)

    x = torch.randn(4, 16, device=device, dtype=torch.float32)
    sigmoid_alpha_out = trptq_gelu.sigmoid_alpha_u8(x, alpha=1.702, exp_lut=lut)
    sigmoid_alpha_expected = sigmoid_alpha_ref(x, lut, alpha=1.702)
    torch.testing.assert_close(sigmoid_alpha_out, sigmoid_alpha_expected, rtol=0, atol=0)

    sigmoid_alpha_mixed_out = trptq_gelu.sigmoid_alpha_mixed_u8(x, alpha=1.702, exp_lut=lut)
    sigmoid_alpha_mixed_expected = sigmoid_alpha_mixed_ref(x, lut, alpha=1.702)
    torch.testing.assert_close(sigmoid_alpha_mixed_out, sigmoid_alpha_mixed_expected, rtol=0, atol=0)

    gelu_out = trptq_gelu.gelu(x, exp_lut=lut)
    gelu_expected = gelu_ref(x, lut)
    torch.testing.assert_close(gelu_out, gelu_expected, rtol=0, atol=0)

    alpha_lut = DEFAULT_ALPHA_TABLE.to(device=device)
    gelu_alpha_out = trptq_gelu.gelu_alpha_table(x, alpha_lut=alpha_lut, exp_lut=lut)
    gelu_alpha_expected = gelu_alpha_table_ref(x, lut, alpha_lut)
    torch.testing.assert_close(gelu_alpha_out, gelu_alpha_expected, rtol=0, atol=0)

    scalar_alpha_lut = torch.full((10,), 1.702, device=device, dtype=torch.float32)
    gelu_alpha_scalar_out = trptq_gelu.gelu_alpha_table(x, alpha_lut=scalar_alpha_lut, exp_lut=lut)
    torch.testing.assert_close(gelu_alpha_scalar_out, gelu_out, rtol=0, atol=0)

    print("sigmoid sample:", sigmoid_out[0, :8].float() / 255.0)
    print("sigmoid mixed sample:", sigmoid_mixed_out[0, :8].float() / 255.0)
    print("sigmoid reciprocal sample:", sigmoid_recip_out[0, :8].float() / 255.0)
    print("sigmoid alpha sample:", sigmoid_alpha_out[0, :8].float() / 255.0)
    print("sigmoid alpha mixed sample:", sigmoid_alpha_mixed_out[0, :8].float() / 255.0)
    print("gelu sample:", gelu_out[0, :8])
    print("gelu alpha-table sample:", gelu_alpha_out[0, :8])
    print("OK")


if __name__ == "__main__":
    main()
