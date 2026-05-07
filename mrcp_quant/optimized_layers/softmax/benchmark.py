import time
import torch
import trptq_softmax_dp4a

A_MIN = -8
A_MAX = 0

def build_lut_u8(device="cuda"):
    anchors = torch.arange(A_MIN, A_MAX + 1, dtype=torch.float32, device=device)
    lut = torch.exp(anchors)
    lut = torch.round(lut / lut.max() * 255.0).to(torch.uint8)
    return lut

def bench_one(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / iters

def main():
    device = "cuda"
    B, N = 512, 256

    x = torch.randn(B, N, device=device)
    x_q44 = torch.round(x * 16.0).clamp(-128, 0).to(torch.int16)

    lut = build_lut_u8(device=device)
    y = torch.zeros_like(x_q44)

    t_dp4a = bench_one(lambda: trptq_softmax_dp4a.softmax_arith_dp4a(x_q44, lut, y))
    print("DP4A kernel:", t_dp4a)

    x_fp = x_q44.float() / 16.0
    t_torch = bench_one(lambda: torch.softmax(x_fp, dim=-1))
    print("PyTorch softmax:", t_torch)

if __name__ == "__main__":
    main()