#!/usr/bin/env python3
"""RTL-vs-model check on the outlier-grid stimulus behind final_error_outlier.csv.
Usage: gen_outlier_rtl_check.py INT_BITS FRAC_W W_MAC. Writes inputs.txt /
expected.txt (GELU, Softmax, RMSNorm per vector, first N_PER_M vectors of each
magnitude, exactly the vectors the CSV is computed from) and meta.txt (M per row)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "scripts", "vit_tiny"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import precision_sweep as ps  # noqa: E402
import precision_analysis_v3 as v3  # noqa: E402
import precision_analysis_v4 as v4  # noqa: E402
import gen_vpu_golden as gg  # noqa: E402
import gen_final_error_tables as g  # noqa: E402

N_PER_M = 60
ib, fw, wmac = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
io_lo, io_hi = -(1 << (ib + fw - 1)), (1 << (ib + fw - 1)) - 1
K = ps._RM_CONST_LN_SQRT_N_Q44
rows, meta, skipped = [], [], 0
for M in g.MAGNITUDES:
    vecs, _ = g.outlier_vectors(M)
    for row in vecs[:N_PER_M]:
        c = np.clip(np.round(row * (1 << fw)), io_lo, io_hi).astype(np.int64)
        y = v3.gelu_precision_v3(c, ib, fw, fw, regions=None, n_anchors_override=8, lut_idx_w=3)
        rows.append((0, c, y, 0)); meta.append(M)
        y, z = v4._softmax_v3_with_z(c, ib, fw, fw, n_anchors_override=8, lut_idx_w=3)
        rows.append((1, c, y, z)); meta.append(M)
        if gg.rmsnorm_is_decay_only(c, ib, fw, K, wmac):
            y = v3.rmsnorm_precision_v3(c, ib, fw, fw, const_value=K, n_anchors_override=8, lut_idx_w=3,
                                        w_mac_trunc_bits=wmac, w_mac_saturate=True, acc_w=32)
            rows.append((2, c, y, 0)); meta.append(M)
        else:
            skipped += 1
open("inputs.txt", "w").write(f"{len(rows)}\n" + "".join(f"{op} " + " ".join(str(int(v)) for v in x) + "\n" for op, x, y, z in rows))
open("expected.txt", "w").write("".join(" ".join(str(int(v)) for v in y) + f" {int(z)}\n" for op, x, y, z in rows))
open("meta.txt", "w").write("\n".join(map(str, meta)) + "\n")
print(f"generated {len(rows)} rows ({skipped} RMSNorm vectors not decay-branch, skipped)")
