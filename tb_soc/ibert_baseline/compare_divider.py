exp = [l.split() for l in open("expected.txt")]
got = [l.split() for l in open("hdl_out.txt")]
if len(exp) != len(got):
    print(f"[FAIL] line count mismatch: exp={len(exp)} got={len(got)}")
    raise SystemExit(1)
mism = 0
first = None
for i, (e, g) in enumerate(zip(exp, got)):
    if e != g:
        mism += 1
        if first is None:
            first = (i, e, g)
status = "PASS" if mism == 0 else "FAIL"
print(f"  [{status}] ibert_divider: {len(exp) - mism}/{len(exp)} exact matches")
if first:
    print(f"    first mismatch row {first[0]}: expected={first[1]} got={first[2]}")
raise SystemExit(0 if mism == 0 else 1)
