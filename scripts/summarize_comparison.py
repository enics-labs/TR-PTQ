#!/usr/bin/env python3
"""Summarize the runs of scripts/run_comparison.sh: copy the reports to reports_cmp/ and write summary.md / summary.json.

usage: summarize_comparison.py --runs <TR-PTQ_runs> --root <TR-PTQ clone> --out <reports_cmp>
Energy per tile = activity-based total power x operation cycles x 3.3 ns (docs/verification.md in the RTL repository).
"""
import argparse, glob, json, os, re, shutil, subprocess, time

T_CLK = 3.3e-9
CYC = {'tr_rmsnorm': 20, 'ibert_rmsnorm': 369, 'tr_swiglu': 30, 'ibert_swiglu': 410, 'tr_softmax': 12, 'ibert_softmax': 208}
VPU_CYC = {'rmsnorm': 15, 'swiglu': 21, 'softmax': 25}
OPS = [('RMSNorm', 'rmsnorm', 'tr_rmsnorm', 'ibert_rmsnorm'), ('SwiGLU', 'swiglu', 'tr_swiglu', 'ibert_swiglu'),
       ('Softmax', 'softmax', 'tr_softmax', 'ibert_softmax')]
DEDICATED = ['tr_rmsnorm', 'ibert_rmsnorm', 'tr_swiglu', 'ibert_swiglu', 'tr_softmax', 'ibert_softmax']
VPU = 'tr_nonlinear_vpu'
def _soc_area(inst, default):
    """Cell area of a top-level instance of tr_soc_top_int from this clone's last top-level run (reports/, written by run_soc_top.sh)."""
    rpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'tr_soc_top_int', 'synthesis', 'post_opt', 'report_area.rpt')
    try:
        for line in open(rpt):
            f = line.split()
            if len(f) >= 4 and f[0] == inst:
                return float(f[3])
    except OSError:
        pass
    return default


CTRL_AREA = _soc_area('u_ctrl', 4335.12)    # u_ctrl of the last tr_soc_top_int run (fallback: d92ec11)
U_VPU_SOC = _soc_area('u_vpu', 46125.72)    # u_vpu of the same run (in-SoC reference)
TAGS_ORDER = ['gated_io10', 'nogate_io10', 'gated_io30']
TAG_TEXT = {'gated_io10': 'clock gating on, boundary delay 10% of the period (main configuration)',
            'nogate_io10': 'clock gating off, boundary delay 10% (diagnostic: sharing effect without gating)',
            'gated_io30': 'clock gating on, boundary delay 30% (the earlier standalone setting; VPU only)'}


def read(p):
    try:
        return open(p, errors='replace').read()
    except OSError:
        return None


def parse_qor(p):
    q = read(p)
    if not q:
        return None
    d = {}
    m = re.search(r'^Cell Area\s+([\d.]+)', q, re.M); d['area'] = float(m.group(1)) if m else None
    for key, rx in (('cells', r'Leaf Instance Count\s+(\d+)'), ('flops', r'Sequential Instance Count\s+(\d+)')):
        m = re.search(rx, q); d[key] = int(m.group(1)) if m else None
    sl = {}
    for g in ('reg2reg', 'in2reg', 'reg2out', 'in2out'):
        m = re.search(r'^' + g + r'\s+(\S+)\s+([\d.]+)\s+(\d+)', q, re.M)
        if m:
            sl[g] = m.group(1)
    d['slack'] = sl
    m = re.search(r'^Total\s+([\d.]+)\s+(\d+)', q, re.M); d['violating'] = int(m.group(2)) if m else None
    return d


def parse_power(p):
    t = read(p)
    if not t:
        return None
    m = re.search(r'Subtotal\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', t)
    if not m:
        return None
    leak, internal, switching, total = (float(x) for x in m.groups())
    r = re.search(r'^\s*register\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', t, re.M)
    return dict(total=total, leak=leak, internal=internal, switching=switching,
                reg_internal=float(r.group(2)) if r else None)


def parse_gating(p):
    t = read(p)
    if not t:
        return None
    g = re.search(r'Total Gated Flip-flops\s+(\d+)', t); u = re.search(r'Total Ungated Flip-flops\s+(\d+)', t)
    if not (g and u):
        return None
    g, u = int(g.group(1)), int(u.group(1))
    return dict(gated=g, ungated=u, frac=g / (g + u) if g + u else 0.0)


def parse_window(p):
    t = read(p)
    if not t:
        return None
    m = re.findall(r'Simulation start = (\d+) ps, end = (\d+) ps', t)
    f = re.search(r'^\s+Flop\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%', t, re.M)
    return dict(window_ns=(int(m[-1][1]) - int(m[-1][0])) / 1000.0 if m else None, flop_annot=float(f.group(5)) if f else None)


def collect(runs, tag, m):
    d = os.path.join(runs, tag)
    P = os.path.join(d, 'reports', m, 'synthesis', 'post_opt')
    qor = parse_qor(os.path.join(P, 'report_qor.rpt'))
    if not qor:
        return None
    out = dict(qor)
    out['gating'] = parse_gating(os.path.join(P, 'report_clock_gating.rpt'))
    vl = parse_power(os.path.join(P, 'report_power.rpt'))
    out['vectorless_total'] = vl['total'] if vl else None
    out['act'] = {}
    sfxs = ['_rmsnorm', '_swiglu', '_softmax'] if m == VPU else ['']
    for s in sfxs:
        pw = parse_power(os.path.join(P, f'report_power_activity{s}.rpt'))
        if not pw:
            continue
        w = parse_window(os.path.join(d, 'workspace', f'{m}_power_activity{s}.log'))
        key = s[1:] if s else ''
        cyc = VPU_CYC[key] if m == VPU else CYC[m]
        pw['cycles'] = cyc
        pw['energy_pj'] = pw['total'] * cyc * T_CLK * 1e12
        if w and w['window_ns']:
            pw['window_ns'] = w['window_ns']
            pw['energy_window_pj'] = pw['total'] * w['window_ns'] * 1e-9 * 1e12
        pw['flop_annot'] = w['flop_annot'] if w else None
        out['act'][key] = pw
    return out


def copy_reports(runs, tag, m, out):
    """Copy the complete report tree of a run (all stages: elaboration, mapping, optimization, export) and its power logs to <out>/<tag>/<m>/."""
    src = os.path.join(runs, tag, 'reports', m)
    dst = os.path.join(out, tag, m)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    for f in glob.glob(os.path.join(runs, tag, 'workspace', f'{m}_power_activity*.log')):
        os.makedirs(dst, exist_ok=True)
        shutil.copy(f, dst)


def f0(x):
    return 'n/a' if x is None else f'{x:,.0f}'


def fp(x, nd=1):
    return 'n/a' if x is None else f'{x:.{nd}f}'


def energy(r, key=''):
    return r['act'][key]['energy_pj'] if r and key in r['act'] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tags = [t for t in TAGS_ORDER if os.path.isdir(os.path.join(a.runs, t))] + \
           sorted(t for t in os.listdir(a.runs) if os.path.isdir(os.path.join(a.runs, t)) and t not in TAGS_ORDER and not t.startswith('_'))
    res = {}
    for tag in tags:
        res[tag] = {}
        for m in DEDICATED + [VPU]:
            r = collect(a.runs, tag, m)
            if r:
                res[tag][m] = r
                copy_reports(a.runs, tag, m, a.out)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)

    rtl = subprocess.run(['git', '-C', os.path.join(a.root, 'tr-vpu_rtl'), 'log', '-1', '--format=%h %s'], capture_output=True, text=True).stdout.strip()
    L = []
    L.append('# Operator comparison: generated results\n')
    L.append('Generated by `scripts/run_comparison.sh` and `scripts/summarize_comparison.py` (synthesis branch `tr-vpu_synthesis`); do not edit by hand. '
             'Raw reports: `reports_cmp/<configuration>/<module>/`.\n')
    L.append(f'- RTL (`tr-vpu_rtl`): {rtl}')
    L.append('- Flow: Genus 25.12-s067_1, `ss_typical_max_0p90v_125c`, clock 3.3 ns, cell area in library units, high effort.')
    L.append('- Energy per tile = activity-based total power x operation cycles x 3.3 ns. Dedicated modules: one 8-element tile (SwiGLU: one 8-lane group, two passes); '
             'VPU: the pass sequence of the same tile (cycles 15 / 21 / 25). Window energy = power x the whole VCD window (includes idle cycles at its end).')
    L.append('- Configurations:')
    for t in tags:
        L.append(f'  - `{t}`: {TAG_TEXT.get(t, "")}')
    L.append('')
    for t in tags:
        L.append(f'## Configuration `{t}`\n')
        L.append('| Module | Cell area | Flops | Gated flops | Slack reg2reg / in2reg / reg2out (ps) | Violating | Cycles | Power (uW) | Energy (pJ) | Window energy (pJ) |')
        L.append('|---|---|---|---|---|---|---|---|---|---|')
        for m in DEDICATED + [VPU]:
            r = res[t].get(m)
            if not r:
                continue
            sl = r['slack']; sl_s = ' / '.join(sl.get(g, 'n/a') for g in ('reg2reg', 'in2reg', 'reg2out'))
            gt = r['gating']['frac'] * 100 if r['gating'] else None
            keys = ['rmsnorm', 'swiglu', 'softmax'] if m == VPU else ['']
            for k in keys:
                p = r['act'].get(k)
                name = f'`{m}` ({k})' if k else f'`{m}`'
                L.append(f"| {name} | {f0(r['area'])} | {r['flops']} | {fp(gt, 0)}% | {sl_s} | {r['violating']} | {p['cycles'] if p else 'n/a'} | "
                         f"{fp(p['total'] * 1e6) if p else 'n/a'} | {fp(p['energy_pj'], 2) if p else 'n/a'} | {fp(p.get('energy_window_pj'), 2) if p else 'n/a'} |")
        L.append('')
        # per operator
        if all(m in res[t] for m in DEDICATED):
            L.append('Per operator (ratios of the table above):\n')
            L.append('| Operator | Area `ibert` / `tr` | Cycles `ibert` / `tr` | Energy `ibert` / `tr` | Area VPU / `tr` sum | Energy VPU / `tr` | Energy VPU / `ibert` |')
            L.append('|---|---|---|---|---|---|---|')
            for name, key, tr, ib in OPS:
                a_t, a_i = res[t][tr], res[t][ib]
                e_t, e_i = energy(a_t), energy(a_i)
                e_v = energy(res[t].get(VPU), key)
                L.append(f"| {name} | {a_i['area'] / a_t['area']:.2f}x | {CYC[ib] / CYC[tr]:.1f}x | {fp(e_i / e_t if e_t and e_i else None, 1)}x | | "
                         f"{fp(e_v / e_t if e_v and e_t else None, 1)}x | {fp(e_v / e_i if e_v and e_i else None, 2)}x |")
            L.append('')
        if VPU in res[t] and all(m in res[t] for m in DEDICATED):
            s_tr = sum(res[t][m]['area'] for _, _, m, _ in OPS); s_ib = sum(res[t][m]['area'] for _, _, _, m in OPS)
            v = res[t][VPU]['area']
            L.append('Sharing in area (the dedicated blocks carry their own sequencing, so the controller `u_ctrl` of the SoC run, %s, is added to the VPU):\n' % f0(CTRL_AREA))
            L.append('| | Cell area | VPU alone | VPU + controller | In-SoC `u_vpu` + controller (%s) |' % f0(U_VPU_SOC + CTRL_AREA))
            L.append('|---|---|---|---|---|')
            for nm, sm in (('Sum of the three `tr_*` blocks', s_tr), ('Sum of the three `ibert_*` blocks', s_ib)):
                L.append(f'| {nm} | {f0(sm)} | {100 * (v / sm - 1):+.1f}% | {100 * ((v + CTRL_AREA) / sm - 1):+.1f}% | {100 * ((U_VPU_SOC + CTRL_AREA) / sm - 1):+.1f}% |')
            L.append(f'\nShared VPU (`{VPU}`): {f0(v)}.\n')
    # effect of clock gating
    if 'gated_io10' in res and 'nogate_io10' in res:
        L.append('## Effect of clock gating (same boundary delay 10%)\n')
        L.append('| Module | Area gated | Area ungated | Gated flops (gated run) | Energy gated (pJ) | Energy ungated (pJ) | Ratio gated / ungated |')
        L.append('|---|---|---|---|---|---|---|')
        for m in DEDICATED + [VPU]:
            g, n = res['gated_io10'].get(m), res['nogate_io10'].get(m)
            if not (g and n):
                continue
            keys = ['rmsnorm', 'swiglu', 'softmax'] if m == VPU else ['']
            for k in keys:
                eg, en = energy(g, k), energy(n, k)
                L.append(f"| `{m}`{' (' + k + ')' if k else ''} | {f0(g['area'])} | {f0(n['area'])} | {fp(g['gating']['frac'] * 100 if g['gating'] else None, 0)}% | "
                         f"{fp(eg, 2)} | {fp(en, 2)} | {fp(eg / en if eg and en else None, 2)} |")
        L.append('')
    if 'gated_io10' in res and 'gated_io30' in res and VPU in res['gated_io30']:
        g10, g30 = res['gated_io10'].get(VPU), res['gated_io30'][VPU]
        if g10:
            L.append('## Effect of the boundary delay on the gated VPU\n')
            L.append(f"Cell area {f0(g30['area'])} at 30% against {f0(g10['area'])} at 10%; slack reg2reg / in2reg / reg2out (30%): "
                     f"{' / '.join(g30['slack'].get(x, 'n/a') for x in ('reg2reg', 'in2reg', 'reg2out'))} ps.\n")
    # earlier results of the dedicated modules (30%, gated) from the main reports
    old = []
    for m in DEDICATED:
        P = os.path.join(a.root, 'reports', m, 'synthesis', 'post_opt')
        q = parse_qor(os.path.join(P, 'report_qor.rpt')); pw = parse_power(os.path.join(P, 'report_power_activity.rpt'))
        if q and pw:
            old.append((m, q['area']))
    if old:
        L.append('## Reference: earlier standalone runs of the dedicated modules (boundary delay 30%, gating on, `reports/`)\n')
        L.append('Cell area only: the activity reports of those runs used pilot VCDs with a 10 ns testbench clock, so their power is not comparable with the energies above.\n')
        L.append('| Module | Cell area |\n|---|---|')
        for m, ar in old:
            L.append(f'| `{m}` | {f0(ar)} |')
        L.append('')
    missing = [(t, m) for t in tags for m in (DEDICATED + [VPU] if t != 'gated_io30' else [VPU]) if m not in res[t]]
    L.append('## Run status\n')
    L.append('Missing synthesis results: ' + (', '.join(f'{t}/{m}' for t, m in missing) if missing else 'none') + '.\n')
    L.append(f'Generated {time.strftime("%Y-%m-%d %H:%M")}.\n')
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(L))
    print('summary written:', os.path.join(a.out, 'summary.md'), '| configurations:', tags, '| missing:', missing)


if __name__ == '__main__':
    main()
