import torch
import trptq_softmax_dp4a

A_MIN = -8
A_MAX = 0

def build_lut_u8(device="cuda"):
    anchors = torch.arange(A_MIN, A_MAX + 1, dtype=torch.float32, device=device)
    lut = torch.exp(anchors)

    # Scale to uint8 max
    lut = torch.round(lut / lut.max() * 255.0).to(torch.uint8)
    return lut

def softmax_ref_arith(x_q44: torch.Tensor, lut_u8: torch.Tensor):
    # x_q44: int16 [B,N]
    x_i32 = x_q44.to(torch.int32)

    row_max = x_i32.max(dim=-1, keepdim=True).values
    z = x_i32 - row_max

    a = (z + 8) >> 4
    a = torch.clamp(a, A_MIN, A_MAX)

    r = z - (a << 4)
    m = 16 + r

    E = lut_u8[(a - A_MIN).long()].to(torch.int32)
    e = (E * m) >> 4

    denom = e.sum(dim=-1, keepdim=True).clamp_min(1)
    y = (e << 8) // denom
    return y.to(torch.int16)

def main():
    device = "cuda"

    #B, N = 4, 128
    B, N = 1, 8
    x = torch.randn(B, N, device=device)
    x = x - x.max()

    print("x:", x)
    # Convert to Q4.4, clamp into a practical range
    x_q44 = torch.round(x * 16.0).clamp(-128, 0).to(torch.int16)
    print("x_q44", x_q44)

    lut = build_lut_u8(device=device)
    print("lut", lut)
    y = torch.zeros_like(x_q44)

    trptq_softmax_dp4a.exp_arith_dp4a(x_q44, lut, out)
    ref = ref_exp(x_q44, lut)
    
    #print("max abs diff:", (y.to(torch.int32) - y_ref.to(torch.int32)).abs().max().item())
    print("sample y[0, :8]:", y[0, :8])
    print("sample ref[0, :8]:", y_ref[0, :8])

if __name__ == "__main__":
    main()
