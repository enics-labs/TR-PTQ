inputs = open("inputs.txt").read().splitlines()[1:]
exp = open("expected.txt").read().splitlines()
got = open("hdl_out.txt").read().splitlines()
meta = [int(x) for x in open("meta.txt").read().split()]
ops = {0: "GELU", 1: "SOFTMAX", 2: "RMSNORM"}
ok = True
print("  %-8s" % "M" + "".join("%18s" % ops[o] for o in (0, 1, 2)))
for M in sorted(set(meta)):
    cells = []
    for o in (0, 1, 2):
        idx = [i for i, l in enumerate(inputs) if meta[i] == M and int(l.split()[0]) == o]
        p = sum(exp[i] == got[i] for i in idx)
        ok &= (p == len(idx))
        cells.append("%d/%d %s" % (p, len(idx), "PASS" if p == len(idx) else "FAIL"))
    print("  %-8d" % M + "".join("%18s" % c for c in cells))
print("  RESULT:", "PASS" if ok else "FAIL")
