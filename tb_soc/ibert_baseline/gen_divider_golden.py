#!/usr/bin/env python3
"""Golden vectors for ibert_divider_tb.sv: random + edge-case (dividend,divisor)
pairs, WIDTH=24 unsigned, with the exact divide-by-zero saturation the RTL
implements (quotient=2^WIDTH-1, remainder=0)."""
import numpy as np

WIDTH = 24
MAXV = (1 << WIDTH) - 1


def golden(dvd, dvs):
    if dvs == 0:
        return MAXV, 0
    return dvd // dvs, dvd % dvs


def main():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(1800):
        dvd = int(rng.integers(0, MAXV + 1))
        dvs = int(rng.integers(0, MAXV + 1))
        rows.append((dvd, dvs))
    # small-operand cases (the actual regime softmax/rmsnorm use)
    for _ in range(200):
        dvd = int(rng.integers(0, 200000))
        dvs = int(rng.integers(0, 6000))
        rows.append((dvd, dvs))
    edges = [(0, 0), (0, 1), (1, 0), (MAXV, 1), (1, MAXV), (MAXV, MAXV),
             (MAXV, 0), (0, MAXV), (5, 5), (7, 3), (1000000, 1)]
    rows += edges

    with open("inputs.txt", "w") as f:
        f.write(f"{len(rows)}\n")
        for dvd, dvs in rows:
            f.write(f"{dvd} {dvs}\n")
    with open("expected.txt", "w") as f:
        for dvd, dvs in rows:
            q, r = golden(dvd, dvs)
            f.write(f"{q} {r}\n")
    print(f"generated {len(rows)} vectors")


if __name__ == "__main__":
    main()
