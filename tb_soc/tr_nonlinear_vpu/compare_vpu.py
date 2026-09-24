import sys

with open("inputs.txt") as f:
    lines = f.readlines()
n = int(lines[0])
inputs = lines[1:1 + n]

exp = [l.split() for l in open("expected.txt")]
hdl = [l.split() for l in open("hdl_out.txt")]
if len(exp) != len(hdl):
    print(f"[FAIL] line count mismatch: exp={len(exp)} hdl={len(hdl)}")
    sys.exit(1)

by_op = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}  # op -> [n_total, n_mismatch]
first_fails = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None}
for i, (inp, e, h) in enumerate(zip(inputs, exp, hdl)):
    op = int(inp.split()[0])
    by_op[op][0] += 1
    if e != h:
        by_op[op][1] += 1
        if first_fails[op] is None:
            first_fails[op] = (i, inp.strip(), e, h)

names = {0: "GELU", 1: "SOFTMAX", 2: "RMSNORM_DECAY", 3: "RMSNORM_GROWTH", 4: "RMSNORM_GROWTH_E<=2 (recip_lut 127-cap)", 5: "RMSNORM_SAT (S>>FRAC_W >= 2^W_MAC)"}
all_pass = True
for op in [0, 1, 2, 3, 4, 5]:
    total, mism = by_op[op]
    if total == 0:
        continue
    status = "PASS" if mism == 0 else "FAIL"
    if mism:
        all_pass = False
    print(f"  [{status}] {names[op]}: {total - mism}/{total} exact matches")
    if mism and first_fails[op]:
        i, inp, e, h = first_fails[op]
        print(f"    first mismatch row {i}: input={inp}")
        print(f"      expected={e}")
        print(f"      got     ={h}")

sys.exit(0 if all_pass else 1)
