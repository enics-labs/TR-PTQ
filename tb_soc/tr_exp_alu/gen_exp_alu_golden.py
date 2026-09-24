#!/usr/bin/env python3
"""Golden generator for tr_exp_alu_tb: exhaustive sweep of all WIDTH-bit
codes, computing (x, hw_final_y, a_idx, is_zero, mantisa) exactly as
round.sv + quadratic_divider.sv (MODE1/generic, matching DECAY_ONLY_LUT=1 /
FRAC_W==4-gated K-map) + tr_exp_alu.sv's ITER=2 padding + the testbench's
own downstream compose (is_zero?255:rom_e_a)*mantisa>>>FRAC_W would."""
import sys

EXP_LUT = [94, 35, 13, 5, 2, 1, 0, 0]  # shared_lut_rom.sv's 8 anchors, format-independent


def golden(width, frac_w, lut_idx_w=3):
    rows = []
    for x in range(-(1 << (width - 1)), 1 << (width - 1)):
        trunc_int = x >> frac_w
        frac_round_bit = (x >> (frac_w - 1)) & 1
        rounded_mag = trunc_int + frac_round_bit
        is_ceil = frac_round_bit
        frac_bits = x & ((1 << frac_w) - 1)
        first_order = ((1 - is_ceil) << frac_w) | frac_bits
        half = 1 << (frac_w - 1)
        delta = frac_bits if frac_bits < half else frac_bits - (1 << frac_w)
        quad_out = (delta * delta) >> (frac_w + 1)
        mantissa = (first_order + quad_out) & ((1 << (frac_w + 2)) - 1)
        is_zero = 1 if rounded_mag == 0 else 0

        lut_max = (1 << lut_idx_w) - 1
        if rounded_mag < -(lut_max + 1):
            a_idx = lut_max  # saturate instead of aliasing; lut_max's anchor is 0
        else:
            low_bits = rounded_mag & ((1 << lut_idx_w) - 1)
            a_idx = (~low_bits) & ((1 << lut_idx_w) - 1)

        e_a = 255 if is_zero else EXP_LUT[a_idx]
        hw_final_y = ((e_a * mantissa) >> frac_w) & ((1 << width) - 1)

        rows.append((x, hw_final_y, a_idx, is_zero, mantissa))
    return rows


if __name__ == "__main__":
    width, frac_w = int(sys.argv[1]), int(sys.argv[2])
    rows = golden(width, frac_w)
    with open("inputs.txt", "w") as f:
        f.write(f"{len(rows)}\n")
        for x, hw_final_y, a_idx, is_zero, mantissa in rows:
            f.write(f"{x}\n")
    with open("expected.txt", "w") as f:
        for x, hw_final_y, a_idx, is_zero, mantissa in rows:
            f.write(f"{x} {hw_final_y} {a_idx} {is_zero} {mantissa}\n")
    print(f"generated {len(rows)} vectors for WIDTH={width} FRAC_W={frac_w}")
