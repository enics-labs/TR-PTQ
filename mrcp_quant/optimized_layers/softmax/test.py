import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import trptq_softmax_dp4a
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent))
    import trptq_softmax_dp4a

from mrcp_quant.quant_utils import new_ln


A_MIN = -8
A_MAX = 0
Q8_U8_A_MIN = -8
Q8_Q16_A_MIN = -16


def softmax_exp_ref_q44(x_q44: torch.Tensor, lut_u8: torch.Tensor):
    x_i32 = x_q44.to(torch.int32)
    a = torch.clamp((x_i32 + 8) >> 4, A_MIN, A_MAX)
    r = x_i32 - (a << 4)
    m = 16 + r
    e = (lut_u8[(a - A_MIN).long()].to(torch.int32) * m) >> 4
    return e.clamp(0, 255).to(torch.uint8)


def gelu_sigmoid_ref_from_q44(x_q44: torch.Tensor, lut_u8: torch.Tensor):
    x_neg = torch.clamp(x_q44, max=0)
    x_pos_neg = (-torch.clamp(x_q44, min=0)).clamp(-128, 0).to(torch.int8)

    exp_int = softmax_exp_ref_q44(x_neg, lut_u8)
    exp_zero = softmax_exp_ref_q44(x_pos_neg, lut_u8)
    exp_sum = exp_int.to(torch.int32) + exp_zero.to(torch.int32)

    ln_sum = new_ln(exp_sum, 8)
    neg_ln_q44 = (-(ln_sum >> 4)).clamp(-128, 0).to(torch.int8)
    ln_mul = softmax_exp_ref_q44(neg_ln_q44, lut_u8)

    sigmoid = ln_mul.to(torch.int32) * exp_int.to(torch.int32)
    return ((sigmoid + 127) // 255).clamp(0, 255).to(torch.uint8)


def exp_q8_u8_ref(z_q8: torch.Tensor):
    lut = torch.tensor([0, 0, 1, 2, 5, 13, 35, 94, 255], device=z_q8.device, dtype=torch.int32)
    z_q8 = z_q8.to(torch.int32)
    a = torch.clamp((z_q8 + 128) >> 8, Q8_U8_A_MIN, 0)
    r_q8 = z_q8 - (a << 8)
    m_q8 = 256 + r_q8
    y = (lut[(a - Q8_U8_A_MIN).long()] * m_q8) >> 8
    return y.clamp(0, 255).to(torch.uint8)


def exp_q8_q16_i32_ref(z_q8: torch.Tensor):
    lut = torch.tensor(
        [0, 0, 0, 0, 0, 1, 3, 8, 22, 60, 162, 442, 1200, 3263, 8869, 24109, 65536],
        device=z_q8.device,
        dtype=torch.int32,
    )
    z_q8 = z_q8.to(torch.int32)
    a = torch.clamp((z_q8 + 128) >> 8, Q8_Q16_A_MIN, 0)
    r_q8 = z_q8 - (a << 8)
    m_q8 = 256 + r_q8
    y = (lut[(a - Q8_Q16_A_MIN).long()] * m_q8) >> 8
    return y.clamp(0, 65536).to(torch.int32)


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
    return torch.where(x > 0, exp_q8_u8_ref(-ln_x_q8), torch.zeros_like(x, dtype=torch.uint8))


def reciprocal_u16_ref(x: torch.Tensor):
    x = x.to(torch.int32)
    ln_x_q8 = new_ln_raw_i32_q8_ref(x)
    y = exp_q8_q16_i32_ref(-ln_x_q8).clamp(0, 65535)
    return torch.where(x > 0, y.to(torch.uint16), torch.zeros_like(x, dtype=torch.uint16))


def reciprocal_q16_i32_ref(x: torch.Tensor):
    x = x.to(torch.int32).clamp(max=1 << 16)
    ln_x_q8 = new_ln(x << 8, 8)
    return torch.where(x > 0, exp_q8_q16_i32_ref(-ln_x_q8), torch.zeros_like(x, dtype=torch.int32))


def main():
    device = "cuda"
    torch.manual_seed(0)

    x_q44 = torch.randint(-128, 1, (4, 16), device=device, dtype=torch.int8)
    lut = trptq_softmax_dp4a.build_exp_lut_u8(device=device)

    exp_out = torch.empty_like(x_q44, dtype=torch.uint8)
    trptq_softmax_dp4a.exp_arith_dp4a(x_q44, lut, exp_out)
    exp_ref = softmax_exp_ref_q44(x_q44, lut)
    torch.testing.assert_close(exp_out, exp_ref, rtol=0, atol=0)

    ln_in = torch.randint(1, 512, (4, 16), device=device, dtype=torch.int32)
    ln_out = torch.empty_like(ln_in)
    trptq_softmax_dp4a.ln_arith(ln_in, 8, ln_out)
    torch.testing.assert_close(ln_out, new_ln(ln_in, 8), rtol=0, atol=0)

    gelu_x = torch.randint(-80, 81, (4, 16), device=device, dtype=torch.int8)
    sigmoid_out = trptq_softmax_dp4a.gelu_sigmoid_u8_from_q44(gelu_x, lut)
    sigmoid_staged = trptq_softmax_dp4a.staged_sigmoid_u8_from_q44(gelu_x, lut)
    sigmoid_ref = gelu_sigmoid_ref_from_q44(gelu_x, lut)
    torch.testing.assert_close(sigmoid_out, sigmoid_staged, rtol=0, atol=0)
    torch.testing.assert_close(sigmoid_out, sigmoid_ref, rtol=0, atol=0)

    recip_u8_x = torch.tensor(
        [0, 1, 2, 16, 17, 32, 255, 256, 1024, 65535, 65536, 1 << 20, 1 << 30, 2147483647],
        device=device,
        dtype=torch.int32,
    )
    recip_u8_out = trptq_softmax_dp4a.reciprocal_u8(recip_u8_x)
    torch.testing.assert_close(recip_u8_out, reciprocal_u8_ref(recip_u8_x), rtol=0, atol=0)

    recip_u16_out = trptq_softmax_dp4a.reciprocal_u16(recip_u8_x)
    torch.testing.assert_close(recip_u16_out, reciprocal_u16_ref(recip_u8_x), rtol=0, atol=0)

    recip_q16_x = torch.tensor([1, 2, 3, 4, 16, 255, 256, 1024, 65535, 65536], device=device, dtype=torch.int32)
    recip_q16_out = trptq_softmax_dp4a.reciprocal_q16_i32(recip_q16_x)
    torch.testing.assert_close(recip_q16_out, reciprocal_q16_i32_ref(recip_q16_x), rtol=0, atol=0)

    sigmoid_deq = sigmoid_out.float() / 255.0
    print("exp sample:", exp_out[0, :8])
    print("ln sample:", ln_out[0, :8])
    print("sigmoid sample:", sigmoid_deq[0, :8])
    print("reciprocal_u8 sample:", recip_u8_out)
    print("reciprocal_u16 sample:", recip_u16_out)
    print("reciprocal_q16_i32 sample:", recip_q16_out)
    print("OK")


if __name__ == "__main__":
    main()
