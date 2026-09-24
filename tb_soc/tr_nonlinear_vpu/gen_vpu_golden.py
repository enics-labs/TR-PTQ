#!/usr/bin/env python3
"""Golden generator for the tr_nonlinear_vpu_tb.sv full-sequence regression
(TEST 5): GELU, Softmax, RMSNorm decay (op 2) and RMSNorm growth (op 3),
native I/O mode, using the precision_analysis_v3.py bit-true model (which
matches the compiled tr_math_model.hpp exactly at Q4.4 -- see
gen_vpu_golden_from_hw.py). Usage: gen_vpu_golden.py INT_BITS FRAC_W [W_MAC].
Writes inputs.txt ("op x0..x7") and expected.txt ("y0..y7 z", z only
meaningful for softmax) in the exact format the testbench reads/writes.
ln(sqrt(N)) constant = ps._RM_CONST_LN_SQRT_N_Q44 (17/16, the as-shipped
value rescaled by LSB shift: codes 17/68/272 at FRAC_W 4/6/8)."""
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


def rmsnorm_ctrl_raw(x_i8, int_bits, frac_w, const_value, w_mac_trunc_bits, saturate=True, acc_w=32):
    """Unclamped, W_VEC-bit-wrapping ctrl_scalar the real controller computes
    (tr_soc_ctrl_int.sv's RM_P3 does a plain W-bit signed add), with the
    same W_MAC saturation of S>>FRAC_W that tr_nonlinear_vpu.sv applies.
    Returns (raw_sum, in_range): vectors whose raw sum would overflow the
    W_VEC-bit register are excluded from RTL comparison (the model clamps,
    the controller wraps -- a controller-side property, not a VPU one)."""
    word_w = int_bits + frac_w
    S = int(np.sum(np.asarray(x_i8, dtype=np.int64) ** 2))
    if acc_w is not None:  # mac_array_engine's signed ACC_W accumulator wraps
        S = ((S + (1 << (acc_w - 1))) % (1 << acc_w)) - (1 << (acc_w - 1))
    shift_amt = 2 * frac_w - frac_w  # native mode: ext_frac_w == frac_w
    sh_signed = S << -shift_amt if shift_amt < 0 else S >> shift_amt
    s_shifted = sh_signed & 0xFFFFFFFFFFFFFFFF
    if w_mac_trunc_bits is not None:
        if saturate:  # rms_s_sat in tr_nonlinear_vpu.sv
            mx = (1 << w_mac_trunc_bits) - 1
            s_shifted = mx if sh_signed > mx else (sh_signed & mx)
        else:
            s_shifted = s_shifted & ((1 << w_mac_trunc_bits) - 1)
    log_s = ps._tr_ln_eval(s_shifted, frac_w, int_bits + frac_w)
    reg_scalar_log = -(log_s >> 1)
    raw_sum = reg_scalar_log + round(const_value * (1 << frac_w))
    lo_w, hi_w = -(1 << (word_w - 1)), (1 << (word_w - 1)) - 1
    return raw_sum, (lo_w <= raw_sum <= hi_w)


def rmsnorm_is_decay_only(x_i8, int_bits, frac_w, const_value, w_mac_trunc_bits=16):
    raw, ok = rmsnorm_ctrl_raw(x_i8, int_bits, frac_w, const_value, w_mac_trunc_bits)
    return ok and raw <= 0


def rmsnorm_is_growth(x_i8, int_bits, frac_w, const_value, w_mac_trunc_bits=16):
    raw, ok = rmsnorm_ctrl_raw(x_i8, int_bits, frac_w, const_value, w_mac_trunc_bits)
    return ok and raw > 0


def gen(int_bits, frac_w, w_mac=16, n_random=200, seed=60):
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
    K = ps._RM_CONST_LN_SQRT_N_Q44

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

    def rms_row(codes):
        return v3.rmsnorm_precision_v3(codes, int_bits, frac_w, ext_frac_w, const_value=K,
                                        n_anchors_override=N_ANCHORS, lut_idx_w=LUT_IDX_W,
                                        w_mac_trunc_bits=w_mac, w_mac_saturate=True, acc_w=32)

    # --- RMSNorm decay (op 2): ctrl_scalar <= 0 ---
    n_got = 0
    tries = 0
    while n_got < n_random and tries < n_random * 50:
        tries += 1
        codes = clip_codes(rng.uniform(-8, 8, size=8))
        if not rmsnorm_is_decay_only(codes, int_bits, frac_w, K, w_mac):
            continue
        rows.append((2, codes, rms_row(codes), 0))
        n_got += 1
    for edge in [np.full(8, io_hi), np.full(8, io_lo)]:
        codes = edge.astype(np.int64)
        if rmsnorm_is_decay_only(codes, int_bits, frac_w, K, w_mac):
            rows.append((2, codes, rms_row(codes), 0))

    # --- RMSNorm growth (op 3): ctrl_scalar > 0 (small-magnitude vectors;
    # log-uniform amplitude so both barely-positive and deep-growth cases occur) ---
    n_got = 0
    tries = 0
    while n_got < n_random and tries < n_random * 400:
        tries += 1
        amp = 10 ** rng.uniform(-1.3, 0.5)
        codes = clip_codes(rng.uniform(-amp, amp, size=8))
        if not rmsnorm_is_growth(codes, int_bits, frac_w, K, w_mac):
            continue
        rows.append((3, codes, rms_row(codes), 0))
        n_got += 1
    for edge in [np.zeros(8, dtype=np.int64), np.ones(8, dtype=np.int64), np.array([1, -1] * 4, dtype=np.int64)]:
        if rmsnorm_is_growth(edge, int_bits, frac_w, K, w_mac):
            rows.append((3, edge, rms_row(edge), 0))

    # --- op 5: RMSNorm vectors whose S>>FRAC_W needs more than W_MAC bits (single
    # outlier + U(-2,2) background). Exists only where the format can reach that
    # (Q8.4/Q12.4 at W_MAC=18); exercises the saturating W_MAC cast in RTL. ---
    if int_bits >= 8 or w_mac < 16:
        n_got = 0
        tries = 0
        while n_got < 100 and tries < 20000:
            tries += 1
            row = rng.uniform(-2, 2, size=8)
            row[rng.integers(0, 8)] = rng.choice([-1.0, 1.0]) * rng.choice([150, 300, 600, 1000])
            codes = clip_codes(row)
            if (int(np.sum(codes ** 2)) >> frac_w) < (1 << w_mac):
                continue
            if not rmsnorm_is_decay_only(codes, int_bits, frac_w, K, w_mac):
                continue
            rows.append((5, codes, rms_row(codes), 0))
            n_got += 1

    return rows


if __name__ == "__main__":
    int_bits, frac_w = int(sys.argv[1]), int(sys.argv[2])
    w_mac = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    rows = gen(int_bits, frac_w, w_mac=w_mac)
    with open("inputs.txt", "w") as f:
        f.write(f"{len(rows)}\n")
        for op, x8, y8, z in rows:
            f.write(f"{op} " + " ".join(str(int(v)) for v in x8) + "\n")
    with open("expected.txt", "w") as f:
        for op, x8, y8, z in rows:
            zval = z if z is not None else 0
            f.write(" ".join(str(int(v)) for v in y8) + f" {int(zval)}\n")
    print(f"generated {len(rows)} vectors (op 0/1/2/3/5) for int_bits={int_bits} frac_w={frac_w} W_MAC={w_mac}")
