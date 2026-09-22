import sys

exp = [l.split() for l in open("expected.txt")]
hdl = [l.split() for l in open("hdl_out.txt")]
if len(exp) != len(hdl):
    print(f"  [FAIL] line count mismatch: exp={len(exp)} hdl={len(hdl)}")
    sys.exit(1)
mism = 0
first_fail = None
for i, (e, h) in enumerate(zip(exp, hdl)):
    if e != h:
        mism += 1
        if first_fail is None:
            first_fail = (i, e, h)
if mism == 0:
    print(f"  [PASS] {len(exp)}/{len(exp)} exact matches")
else:
    print(f"  [FAIL] {mism}/{len(exp)} mismatches, first at row {first_fail[0]}: expected={first_fail[1]} got={first_fail[2]}")
