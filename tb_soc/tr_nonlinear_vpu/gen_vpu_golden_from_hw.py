#!/usr/bin/env python3
"""Q4.4-only functional check generator: writes inputs.txt/expected.txt using
the REAL compiled hardware model (tools/infra/tr_math_hw.py -> the exact
tr_math_model.hpp code cpu_math_model.cpp/verify_block.py checks real RTL
against), NOT the Python tr_vpu_model approximation used by
gen_vpu_golden.py. RMSNorm decay (op 2) uses rmsnorm_hw(); growth (op 3)
uses rmsnorm_hw_signguard_fixed(), the model of the production FSM's
sign-guard + recip_lut path (plain rmsnorm_hw() is the pre-fix hardware and
collapses on growth vectors). NOTE the compiled model is Q4.4/int8-only, so
this check exists at Q4.4 only. Purpose: an end-to-end check that the newly-parameterized
tr_nonlinear_vpu.sv, at its Q4.4 defaults, still produces bit-identical
output to the original as-shipped hardware model -- independent of whatever
the precision-sweep Python model claims."""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
import infra.tr_math_hw as hw  # noqa: E402


def rmsnorm_is_decay_only(x_i8):
    """Mirror of gen_vpu_golden.py's filter, but using the SAME constant
    (17, RMSNORM_CONST_LN_SQRT_N in tr_math_model.hpp) and asking the real
    hw model's own ln pass for reg_scalar_log, so the filter and the
    expected-output source are both grounded in the same compiled model."""
    S = int(np.sum(np.asarray(x_i8, dtype=np.int64) ** 2))
    s_shifted = (S >> 4) & 0xFFFF  # native Q4.4: shift_amt = 2*4-4 = 4; W_MAC=16 truncation
    log_s = hw.rmsnorm_ln_pass(s_shifted)
    reg_scalar_log = -(log_s >> 1)
    ctrl_scalar = reg_scalar_log + 17
    word_w = 8
    lo_w, hi_w = -(1 << (word_w - 1)), (1 << (word_w - 1)) - 1
    if ctrl_scalar < lo_w or ctrl_scalar > hi_w:
        return False
    return ctrl_scalar <= 0


def rmsnorm_is_growth(x_i8):
    S = int(np.sum(np.asarray(x_i8, dtype=np.int64) ** 2))
    s_shifted = (S >> 4) & 0xFFFF
    log_s = hw.rmsnorm_ln_pass(s_shifted)
    reg_scalar_log = -(log_s >> 1)
    ctrl_scalar = reg_scalar_log + 17
    return 0 < ctrl_scalar <= 127


def growth_expected(codes):
    """(expected_out, E, exact_vs_compiled). Compiled rmsnorm_hw_signguard_fixed()
    caps inv_rms at 255; the production controller's recip_lut() caps at 127.
    They only differ when E<=2 (256/E>127). For those, rebuild the compiled
    model's arithmetic from its own primitives with the controller's 127 cap."""
    S = int(np.sum(codes.astype(np.int64) ** 2))
    reg_scalar_log = -(hw.rmsnorm_ln_pass((S >> 4) & 0xFFFF) >> 1)
    E = hw.rmsnorm_exp_pass(-(reg_scalar_log + 17))
    fixed = hw.rmsnorm_hw_signguard_fixed(codes)
    if E >= 3:
        return fixed, E, True
    inv = min(127, (256 + max(E, 1) // 2) // max(E, 1))
    out = np.array([np.int8((int(x) * inv) >> 4) for x in codes], dtype=np.int8)
    return out, E, False


def check_recip_lut_matches_tb():
    """The tb's recip_ideal() (round(256/max(E,1)) sat 127) must equal the
    controller's actual recip_lut() for every E it can see (0..31)."""
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "..", "rtl_soc", "tr_soc_ctrl", "tr_soc_ctrl_int.sv")).read()
    body = src[src.index("function automatic logic signed [W-1:0] recip_lut"):]
    body = body[:body.index("endfunction")]
    lut = {}
    for m in re.finditer(r"8'sd(\d+)(?:,\s*8'sd(\d+))?:\s*recip_lut = 8'sd(\d+);", body):
        for k in (m.group(1), m.group(2)):
            if k is not None:
                lut[int(k)] = int(m.group(3))
    bad = []
    for e in range(0, 32):
        ideal = min(127, (256 * 2 + max(e, 1)) // (2 * max(e, 1)))  # round-half-up == round-half-even here (no ties)
        if lut.get(e) != ideal:
            bad.append((e, lut.get(e), ideal))
    print("recip_lut (parsed from tr_soc_ctrl_int.sv) vs tb recip_ideal, E=0..31:",
          "MATCH" if not bad else f"MISMATCH {bad}")
    assert not bad


def gen(n_random=200, seed=61):
    io_lo, io_hi = -128, 127
    rng = np.random.default_rng(seed)
    rows = []  # (op, x8, y8, z)

    # --- GELU (op 0) ---
    for _ in range(n_random):
        row = rng.uniform(-8, 8, size=8)
        codes = np.clip(np.round(row * 16), io_lo, io_hi).astype(np.int8)
        y = hw.gelu_hw(codes)
        rows.append((0, codes, y, 0))
    for edge in [np.full(8, io_hi, dtype=np.int8), np.full(8, io_lo, dtype=np.int8), np.zeros(8, dtype=np.int8)]:
        y = hw.gelu_hw(edge)
        rows.append((0, edge, y, 0))

    # --- Softmax (op 1) ---
    for _ in range(n_random):
        row = rng.uniform(-8, 8, size=8)
        codes = np.clip(np.round(row * 16), io_lo, io_hi).astype(np.int8)
        y, z = hw.softmax_hw_ex(codes)
        rows.append((1, codes, y, z))
    for edge in [np.full(8, io_hi, dtype=np.int8), np.full(8, io_lo, dtype=np.int8), np.zeros(8, dtype=np.int8)]:
        y, z = hw.softmax_hw_ex(edge)
        rows.append((1, edge, y, z))

    # --- RMSNorm decay (op 2), real rmsnorm_hw() ---
    n_needed = n_random
    n_got = 0
    tries = 0
    while n_got < n_needed and tries < n_needed * 50:
        tries += 1
        row = rng.uniform(-8, 8, size=8)
        codes = np.clip(np.round(row * 16), io_lo, io_hi).astype(np.int8)
        if not rmsnorm_is_decay_only(codes):
            continue
        y = hw.rmsnorm_hw(codes)
        rows.append((2, codes, y, 0))
        n_got += 1
    for edge in [np.full(8, io_hi, dtype=np.int8), np.full(8, io_lo, dtype=np.int8)]:
        if rmsnorm_is_decay_only(edge):
            y = hw.rmsnorm_hw(edge)
            rows.append((2, edge, y, 0))

    # --- RMSNorm growth (op 3), rmsnorm_hw_signguard_fixed() ---
    n_got = 0
    tries = 0
    while n_got < n_random and tries < n_random * 400:
        tries += 1
        amp = 10 ** rng.uniform(-1.3, 0.5)
        codes = np.clip(np.round(rng.uniform(-amp, amp, size=8) * 16), io_lo, io_hi).astype(np.int8)
        if not rmsnorm_is_growth(codes):
            continue
        out, E, exact = growth_expected(codes)
        rows.append((3 if exact else 4, codes, out, 0))
        n_got += 1
    for edge in [np.zeros(8, dtype=np.int8), np.ones(8, dtype=np.int8), np.array([1, -1] * 4, dtype=np.int8)]:
        if rmsnorm_is_growth(edge):
            out, E, exact = growth_expected(edge)
            rows.append((3 if exact else 4, edge, out, 0))

    # decay vectors: fixed and unfixed compiled models must agree (guard is a no-op there)
    n_dis = sum(1 for op, x, y, z in rows if op == 2 and not np.array_equal(hw.rmsnorm_hw_signguard_fixed(x), y))
    print(f"decay vectors where rmsnorm_hw != rmsnorm_hw_signguard_fixed: {n_dis}")
    n3 = sum(1 for r in rows if r[0] == 3)
    n4 = sum(1 for r in rows if r[0] == 4)
    print(f"growth vectors: {n3} with E>=3 (expected = compiled model verbatim), "
          f"{n4} with E<=2 (compiled model caps inv_rms at 255, controller recip_lut at 127; expected uses 127)")
    return rows


if __name__ == "__main__":
    check_recip_lut_matches_tb()
    rows = gen()
    with open("inputs.txt", "w") as f:
        f.write(f"{len(rows)}\n")
        for op, x8, y8, z in rows:
            f.write(f"{op} " + " ".join(str(int(v)) for v in x8) + "\n")
    with open("expected.txt", "w") as f:
        for op, x8, y8, z in rows:
            zval = z if z is not None else 0
            f.write(" ".join(str(int(v)) for v in y8) + f" {int(zval)}\n")
    print(f"generated {len(rows)} vectors (op 0/1/2/3/4) from the REAL hw model (tr_math_hw.py)")
