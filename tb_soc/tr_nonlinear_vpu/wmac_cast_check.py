inputs = open("inputs.txt").read().splitlines()[1:]
exp = open("expected.txt").read().splitlines()
pre = open("hdl_out_pre.txt").read().splitlines()
post = open("hdl_out_post.txt").read().splitlines()
names = {0: "GELU", 1: "SOFTMAX", 2: "RMSNORM_DECAY", 3: "RMSNORM_GROWTH", 5: "RMSNORM_SAT"}
ok = True
print("  %-15s %4s | patched==golden | wrap-copy==golden | patched==wrap-copy" % ("op", "n"))
for op in (0, 1, 2, 3, 5):
    idx = [i for i, l in enumerate(inputs) if int(l.split()[0]) == op]
    if not idx:
        continue
    a = sum(post[i] == exp[i] for i in idx)
    b = sum(pre[i] == exp[i] for i in idx)
    c = sum(post[i] == pre[i] for i in idx)
    n = len(idx)
    print("  %-15s %4d |   %4d/%-4d    |    %4d/%-4d      |    %4d/%-4d" % (names[op], n, a, n, b, n, c, n))
    ok &= (a == n)                                  # patched RTL always equals the saturating golden
    if op == 5:
        ok &= (b < n)                               # the wrap version must actually differ here
    elif op in (0, 1, 3):
        ok &= (c == n)                              # untouched ops: byte-identical before/after
print("  RESULT:", "PASS" if ok else "FAIL")
