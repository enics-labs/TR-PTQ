#!/usr/bin/env python3
"""Bit-true Python port of ibert_rmsnorm.sv (Q4.4, N=8, W=8, FRAC_W=4)."""
import numpy as np

N = 8
W = 8
MAX_SQRT_ITERS = 16


def i_sqrt(n):
    """I-BERT Algorithm 4, adapted (see ibert_rmsnorm.sv header)."""
    if n == 0:
        return 0
    bits = n.bit_length()
    ceil_half = (bits + 1) >> 1
    x = 1 << ceil_half
    for _ in range(MAX_SQRT_ITERS):
        q = n // x
        x_next = (x + q) >> 1
        if x_next >= x:
            return x
        x = x_next
    return x  # safety-cap fallback, matches the RTL


def golden(x_in):
    x = np.asarray(x_in, dtype=np.int64)
    sum_sq = int(np.sum(x * x))
    mean_sq = sum_sq >> 3
    rms_code = i_sqrt(mean_sq)
    if rms_code == 0:
        return [0] * N
    out = []
    for xi in x_in:
        abs_x = abs(int(xi))
        dividend = (abs_x << 4) + (rms_code >> 1)  # <<FRAC_W (4), not <<8: see ibert_rmsnorm.sv
        q = dividend // rms_code
        q_clip = min(q, (1 << (W - 1)) - 1)
        out.append(-q_clip if xi < 0 else q_clip)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    rows = []
    for _ in range(500):
        rows.append(rng.integers(-128, 128, size=N).tolist())
    edges = [
        [0] * N, [127] * N, [-128] * N,
        [1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0],
        [127, -128, 0, 0, 0, 0, 0, 0],
        [1] * N, [-1] * N,
        [2, -2, 2, -2, 2, -2, 2, -2],
        [127, 1, 1, 1, 1, 1, 1, 1],
    ]
    rows += edges

    with open("inputs.txt", "w") as f:
        f.write(f"{len(rows)}\n")
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")
    with open("expected.txt", "w") as f:
        for row in rows:
            f.write(" ".join(map(str, golden(row))) + "\n")
    print(f"generated {len(rows)} vectors")
