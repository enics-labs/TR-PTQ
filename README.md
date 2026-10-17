# TR-PTQ VPU: synthesis

Cadence Genus synthesis flow (65 nm TSMC/ARM standard cells) for the TR-PTQ VPU RTL.
This is `tr-vpu_synthesis`, the synthesis-only branch of [TR-PTQ](https://github.com/enics-labs/TR-PTQ).
The RTL lives on the `tr-vpu_rtl` branch and is pulled in here as the `tr-vpu_rtl/` submodule.

Reports of the paper-era designs (I-BERT and standalone TR baselines, MX top, bit-width sweeps) are not kept
in this branch; they are preserved at the tag `tr-vpu_synthesis-ISCAS-20261013`.

## Current results

Default build (no GELU), clock 3.3 ns, `ss_typical_max_0p90v_125c`. Cell area in library units.

| Top level | RTL commit | Timing | Cell area | Notes |
|---|---|---|---|---|
| `tr_soc_top_int` | `56c819e` | met, 0 violating paths | 178,000 | `u_vpu` 46,706; power 33.3 mW (vectorless) |
| `tr_nonlinear_vpu` (standalone) | `e65eb37` | in2reg -10 ps | 56,721 | the violating path is an input-delay artifact of standalone runs |
| `tr_backbone_wrapper` (standalone) | `2c7ce3a` | met | 26,760 | before the GELU-free default; includes all 8 ln lanes |

Raw reports: `reports/<top level>/synthesis/`. Analysis and history: `docs/llama/synthesis-results.md`
in the [integration repository](https://github.com/YoniP31/tr-ptq-ddp23).

## Layout

| Path | Content |
|---|---|
| `tr-vpu_rtl/` | submodule: the RTL (only `rtl_soc/` and `rtl_baseline/` are read) |
| `inputs/dut.defines`, `inputs/dut.sdc` | technology, clock period (3.3 ns), I/O delays, DRV limits, path layout, RTL roots |
| `libraries/` | TSMC65LP / ARM65LP standard-cell, SRAM and IO library configuration |
| `scripts/` | `genus.tcl` (main flow), `report_power.tcl`, `report_power_activity.tcl`, `run_sweep.tcl`, `procedures.tcl` |
| `reports/<top>/synthesis/` | timing, area, power, QoR reports of the last run |
| `export/<top>/` | exported netlist, db, sdf, sdc (git-ignored) |
| `workspace/` | run everything from here (tracked empty; all generated files are untracked) |

## Setup

- Cadence Genus, with a license server reachable through `CDS_LIC_FILE`.
- `tcsh` (the commands below use its syntax).
- The RTL submodule: `git submodule update --init --recursive`.
  It does not follow its branch automatically; after RTL changes run `git submodule update --remote tr-vpu_rtl`.

## Running

All commands run from `workspace/`.

```tcsh
cd workspace
genus -execute \"set TOPLEVEL <module>\" -f ../scripts/genus.tcl -log <module>_synth
```

- `<module>` is any module under `tr-vpu_rtl/rtl_soc/` or `tr-vpu_rtl/rtl_baseline/`
  (typically `tr_soc_top_int`, `tr_nonlinear_vpu`, `tr_backbone_wrapper`).
  `genus.tcl` reads every `.sv` file in both trees and elaborates only the module you name.
- The escaped quotes are required for `-execute` in this environment.
- The flow reads the libraries and RTL, synthesizes (generic, mapped, optimized) at high effort, exports the
  design to `export/<module>/post_synth/`, and then writes the reports.
- `genus.tcl` takes `TOPLEVEL` from the environment first. A leftover `setenv TOPLEVEL ...` in your shell silently
  overrides `-execute`; run `unsetenv TOPLEVEL` before starting a new run.
- Reports are named by top level only, so a second run of the same module overwrites the first.
  Rename `reports/<module>/synthesis` before re-running if you want to keep it.

### Build flavors

The RTL has a compile-time `GELU_EN` switch. The default build has no GELU (the Llama configuration). To synthesize
the GELU build, add `-define GELU_ENABLE` to the `read_hdl` call in `genus.tcl`, or elaborate a standalone block
with `-parameters {{GELU_EN 1}}`. See the RTL branch documentation for what the switch removes.

### Power

Vectorless estimate on an already-synthesized design (default 12.5% toggle rate, takes seconds):

```tcsh
genus -execute \"set TOPLEVEL <module>\" -f ../scripts/report_power.tcl -log <module>_power
```

Activity-based power from a VCD of an `*_energy_tb` pilot testbench (RTL branch):

```tcsh
setenv ACTIVITY_VCD <path>.vcd
setenv ACTIVITY_INSTANCE <testbench_module>/<dut_instance>
genus -execute \"set TOPLEVEL <module>\" -f ../scripts/report_power_activity.tcl -log <module>_power_activity
```

Energy = total power x cycles x clock period (3.3 ns). Both scripts write one fixed file name per top level
(`report_power.rpt`, `report_power_activity.rpt`); keep a copy if you need the previous result.
Set the three environment variables separately; combining them in one `-execute` string breaks the CLI here.

### Fmax sweep

`run_sweep.tcl` walks the clock period down from `START_PERIOD` to `END_PERIOD` in `STEP` ns steps at low effort,
stops at the first period with negative slack or a DRV violation, then re-synthesizes the last clean period at full
effort.

```tcsh
setenv START_PERIOD 10
setenv END_PERIOD 2
setenv STEP 1
genus -execute \"set TOPLEVEL <module>\" -f ../scripts/run_sweep.tcl -log <module>_sweep
```

Defaults are 10 / 2 / 1 ns. Results: `reports/<module>/sweep/clock_sweep_results.csv` and
`export/<module>/post_synth/<module>.sweep_optimal.db`.

## Reading a result

- Genus stops optimizing a group at zero slack, so "met" reports show 0 to a few ps of slack, not margin.
  Use the sweep to find the real limit.
- Standalone runs of inner blocks constrain every input and output with 30% of the clock as delay. Paths that start
  at a port which is a register output in the full design can then show violations that do not exist at the top level.
  Judge timing on the top level.
- The area of a block depends on the timing pressure on it. A block that sits on a failing path is upsized and
  duplicated; compare areas only between runs that all meet timing.

## Limitations

- `scripts/settings.tcl` and `scripts/simulation.tcl` (OCV settings, gate-level VCD probing) are not used by the flow.
- There is no place-and-route, back-annotated gate-level simulation or power-rail analysis in this branch.
