#!/usr/bin/env python3
"""Golden generator for the tr_nonlinear_vpu_tb.sv full-sequence regression
(TEST 5): GELU, Softmax, and RMSNorm's decay-only case, native I/O mode,
using the already-validated precision_analysis_v3.py model. Writes
inputs.txt ("op x0..x7") and expected.txt ("y0..y7 z", z only meaningful
for softmax) in the exact format the testbench reads/writes."""
import math
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "scripts", "vit_tiny"))

import numpy as np  # noqa: E402
import precision_sweep as ps  # noqa: E402
import precision_analysis as pa  # noqa: E402
import precision_analysis_v3 as v3  # noqa: E402
import precision_analysis_v4 as v4  # noqa: E402


def rmsnorm_is_decay_only(x_i8, int_bits, frac_w, const_value, w_mac_trunc_bits=16):
    """True only if ctrl_scalar = reg_scalar_log + const is BOTH <=0 AND
    doesn't overflow the W_VEC-bit register on the way there. The real
    controller (tr_soc_ctrl_int.sv's RM_P3) does a plain W-bit signed add
    with no saturation -- unlike rmsnorm_precision_v3's own clamped
    ctrl_scalar, which would silently disagree with the real (wrapping)
    hardware for the rare overflowing case. That's a pre-existing
    controller-side characteristic, not something introduced by
    tr_nonlinear_vpu.sv's parameterization, so this test simply excludes
    those vectors rather than trying to model the wrap."""
    word_w = int_bits + frac_w
    scale = 1 << frac_w
    S = int(np.sum(np.asarray(x_i8, dtype=np.int64) ** 2))
    shift_amt = 2 * frac_w - frac_w  # native mode: ext_frac_w == frac_w
    s_shifted = (S << -shift_amt if shift_amt < 0 else S >> shift_amt) & 0xFFFFFFFFFFFFFFFF
    if w_mac_trunc_bits is not None:
        s_shifted = s_shifted & ((1 << w_mac_trunc_bits) - 1)  # matches W_MAC'(...) truncation in tr_nonlinear_vpu.sv
    log_s = ps._tr_ln_eval(s_shifted, frac_w, int_bits + frac_w)
    reg_scalar_log = -(log_s >> 1)
    const_code = round(const_value * scale)
    raw_sum = reg_scalar_log + const_code  # UNCLAMPED, matching the RTL's raw add
    lo_w, hi_w = -(1 << (word_w - 1)), (1 << (word_w - 1)) - 1
    if raw_sum < lo_w or raw_sum > hi_w:
        return False  # would overflow/wrap the W_VEC-bit register -- out of scope
    return raw_sum <= 0


def gen(int_bits, frac_w, n_random=200, seed=60):
    ext_frac_w = frac_w
    io_lo, io_hi = -(1 << (int_bits + ext_frac_w - 1)), (1 << (int_bits + ext_frac_w - 1)) - 1
    rng = np.random.default_rng(seed)

    rows = []  # (op, x8, y8, z)

    def clip_codes(row_real):
        return np.clip(np.round(row_real * (1 << ext_frac_w)), io_lo, io_hi).astype(np.int64)

    # n_anchors AND lut_idx_w both forced to match shared_lut_rom.sv exactly
    # (8 entries, LUT_IDX_W=3, for all 6 formats): alpha_stabilizer's own
    # saturation bound widens with W_VEC, so alpha itself (an internal,
    # derived value, not x directly) CAN exceed magnitude 8 once int_bits
    # grows -- exercising anchor indices beyond the real 8-entry table's
    # range, which real hardware handles by WRAPPING the index (round.sv's
    # ~rounded_mag[LUT_IDX_W-1:0]), not by returning 0. Without lut_idx_w
    # set, _tr_exp_eval_raw's default behavior (clamp any out-of-range
    # magnitude to e_a=0) does NOT match that wrap and was caught
    # disagreeing with real tr_nonlinear_vpu RTL simulation directly (a
    # bug in this Python model, not the RTL). precision_analysis_v3.py's
    # default n_anchors (up to 15 for wider int_bits) is a DIFFERENT,
    # wider-anchor-table design point from earlier analysis work -- not
    # what shared_lut_rom.sv actually implements, so both must be pinned
    # here to match the real RTL.
    N_ANCHORS = 8
    LUT_IDX_W = 3

    # --- GELU (op 0) ---
    for _ in range(n_random):
        row = rng.uniform(-8, 8, size=8)
        codes = clip_codes(row)
        y = v3.gelu_precision_v3(codes, int_bits, frac_w, ext_frac_w, regions=None, n_anchors_override=N_ANCHORS, lut_idx_w=LUT_IDX_W)
        rows.append((0, codes, y, 0))
    # edge cases
    for edge in [np.full(8, io_hi), np.full(8, io_lo), np.zeros(8, dtype=np.int64)]:
        codes = edge.astype(np.int64)
        y = v3.gelu_precision_v3(codes, int_bits, frac_w, ext_frac_w, regions=None, n_anchors_override=N_ANCHORS, lut_idx_w=LUT_IDX_W)
        rows.append((0, codes, y, 0))

    # --- Softmax (op 1) ---
    for _ in range(n_random):
        row = rng.uniform(-8, 8, size=8)
        codes = clip_codes(row)
        y, z = v4._softmax_v3_with_z(codes, int_bits, frac_w, ext_frac_w, n_anchors_override=N_ANCHORS, lut_idx_w=LUT_IDX_W)
        rows.append((1, codes, y, z))
    for edge in [np.full(8, io_hi), np.full(8, io_lo), np.zeros(8, dtype=np.int64)]:
        codes = edge.astype(np.int64)
        y, z = v4._softmax_v3_with_z(codes, int_bits, frac_w, ext_frac_w, n_anchors_override=N_ANCHORS, lut_idx_w=LUT_IDX_W)
        rows.append((1, codes, y, z))

    # --- RMSNorm decay-only (op 2) -- filter for ctrl_scalar<=0 ---
    n_needed = n_random
    n_got = 0
    tries = 0
    while n_got < n_needed and tries < n_needed * 50:
        tries += 1
        row = rng.uniform(-8, 8, size=8)
        codes = clip_codes(row)
        if not rmsnorm_is_decay_only(codes, int_bits, frac_w, v3._RM_CONST_TRUE):
            continue
        y = v3.rmsnorm_precision_v3(codes, int_bits, frac_w, ext_frac_w, const_value=v3._RM_CONST_TRUE, n_anchors_override=N_ANCHORS, lut_idx_w=LUT_IDX_W, w_mac_trunc_bits=16)
        rows.append((2, codes, y, 0))
        n_got += 1
    # edge cases likely to be decay-only (large-magnitude, so S is large)
    for edge in [np.full(8, io_hi), np.full(8, io_lo)]:
        codes = edge.astype(np.int64)
        if rmsnorm_is_decay_only(codes, int_bits, frac_w, v3._RM_CONST_TRUE):
            y = v3.rmsnorm_precision_v3(codes, int_bits, frac_w, ext_frac_w, const_value=v3._RM_CONST_TRUE, n_anchors_override=N_ANCHORS, lut_idx_w=LUT_IDX_W, w_mac_trunc_bits=16)
            rows.append((2, codes, y, 0))

    return rows


if __name__ == "__main__":
    int_bits, frac_w = int(sys.argv[1]), int(sys.argv[2])
    rows = gen(int_bits, frac_w)
    with open("inputs.txt", "w") as f:
        f.write(f"{len(rows)}\n")
        for op, x8, y8, z in rows:
            f.write(f"{op} " + " ".join(str(int(v)) for v in x8) + "\n")
    with open("expected.txt", "w") as f:
        for op, x8, y8, z in rows:
            zval = z if z is not None else 0
            f.write(" ".join(str(int(v)) for v in y8) + f" {int(zval)}\n")
    print(f"generated {len(rows)} vectors (op 0/1/2) for int_bits={int_bits} frac_w={frac_w}")
