#!/usr/bin/env python3
"""Bit-true Python port of ibert_softmax.sv (Q4.4, N=8, W=8, FRAC_W=4).
Random + edge-case 8-wide vectors."""
import numpy as np

N = 8
QLN2 = 11
QB = 21
QC = 245


def golden(x):
    x = np.asarray(x, dtype=np.int64)
    max_x = int(np.max(x))
    exp_code = []
    for xi in x:
        diff = int(xi) - max_x           # <= 0
        z = (-diff) // QLN2
        qp = diff + z * QLN2
        delta = qp + QB
        delta_sq = delta * delta
        qout = delta_sq + QC
        exp_code.append(qout >> z)
    total = sum(exp_code)
    out = []
    for e in exp_code:
        dividend = (e << 8) + (total >> 1)
        q = dividend // total
        out.append(min(255, q))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(400):
        rows.append(rng.integers(-128, 128, size=N).tolist())
    edges = [
        [127] * N, [-128] * N, [0] * N,
        [127, -128, 0, 64, -64, 32, -32, 1],
        [127] + [-128] * (N - 1),
        [0, 1, -1, 2, -2, 3, -3, 4],
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
