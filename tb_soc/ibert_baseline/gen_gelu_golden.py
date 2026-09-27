#!/usr/bin/env python3
"""Bit-true Python port of ibert_gelu.sv (Q4.4, W=8, FRAC_W=4). Two parts:
exhaustive sweep of all 256 int8 codes (same code broadcast to every lane,
checked on lane 0) plus a random pass with independent per-lane codes
(checked on all 8 lanes, to catch any cross-lane coupling bug). Ports the
RTL's exact fixed-point sequence, not the paper's abstract real-scale
Algorithm 1/2 (see ibert_gelu.sv's header)."""
import numpy as np

N = 8
K1, K1_SHIFT = 2897, 8
CLIP_T = 453
B_CODE = -453
A_CODE = -74
A_SHIFT = 16
ONE_Q8 = 256
Y_SHIFT = 9


def round_shift(val, shift):
    return (val + (1 << (shift - 1))) >> shift


def golden(x_code):
    t_code = round_shift(x_code * K1, K1_SHIFT)
    qsgn = t_code < 0
    q_abs = min(abs(t_code), CLIP_T)
    delta = q_abs + B_CODE
    delta_sq = delta * delta
    erf_partial = round_shift(A_CODE * delta_sq, A_SHIFT)
    erf_pos = erf_partial + ONE_Q8
    erf = -erf_pos if qsgn else erf_pos
    one_plus_erf = erf + ONE_Q8
    y_raw = round_shift(x_code * one_plus_erf, Y_SHIFT)
    return max(-128, min(127, y_raw))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    exhaustive = list(range(-128, 128))
    n_random = 300
    random_rows = [rng.integers(-128, 128, size=N).tolist() for _ in range(n_random)]

    with open("inputs.txt", "w") as f:
        f.write(f"{len(exhaustive) + len(random_rows)}\n")
        for x in exhaustive:
            f.write(f"E {x}\n")
        for row in random_rows:
            f.write("R " + " ".join(map(str, row)) + "\n")
    with open("expected.txt", "w") as f:
        for x in exhaustive:
            f.write(f"{golden(x)}\n")
        for row in random_rows:
            f.write(" ".join(str(golden(x)) for x in row) + "\n")
    print(f"generated {len(exhaustive)} exhaustive + {len(random_rows)} random vectors")
