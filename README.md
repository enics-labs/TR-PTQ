# TR-PTQ VPU — Synthesis

Cadence Genus (65nm TSMC/ARM standard cells) synthesis flow for the TR-PTQ VPU RTL. This is `tr-vpu_synthesis`, the synthesis-only branch of [TR-PTQ](https://github.com/enics-labs/TR-PTQ). The RTL itself lives on the `tr-vpu_rtl` branch (pulled in here as a submodule, see below) — see that branch's `docs/architecture.md` for what's actually being synthesized.

## Directory structure

```
tr-vpu_rtl/             Git submodule -- the tr-vpu_rtl branch of this same
                         repo. Only rtl_soc/ and rtl_baseline/ are used by
                         this flow; everything else in there (tb_soc/,
                         docs/, etc.) is unused but present.
inputs/
  dut.defines            Shared settings: technology, clock period (3.3 ns),
                          I/O delays, DRV constraints, file/path layout,
                          and rtl_dirs (the two RTL roots read below).
  dut.sdc                 Shared SDC template (clocks, I/O delay, load/drive,
                          max fanout/transition) -- driven by dut.defines' values.
libraries/              TSMC65LP / ARM65LP standard-cell, SRAM, and IO lib configs.
scripts/
  genus.tcl               Main synthesis flow (read -> elaborate -> synthesize -> export -> reports).
  report_power.tcl         Vectorless power report on an already-synthesized design.
  report_power_activity.tcl  VCD-driven (real activity) power report.
  run_sweep.tcl            Fmax sweep (see below).
  procedures.tcl           Shared helper procs, sourced by the scripts above.
reports/<TOPLEVEL>/     Synthesis output: timing, area, power, QoR reports.
export/<TOPLEVEL>/      Exported netlist/db/sdf/sdc from a completed synthesis run.
workspace/              Run everything from here. Tracked as an empty directory
                          (.gitkeep) -- everything generated inside stays untracked.
```

## Where the RTL comes from

`tr-vpu_rtl/` is a **git submodule** pointing at the `tr-vpu_rtl` branch of this same repo (`enics-labs/TR-PTQ`). First-time setup:

```sh
git submodule update --init --recursive
```

`genus.tcl` reads **every** `.sv` file under `tr-vpu_rtl/rtl_soc/` and `tr-vpu_rtl/rtl_baseline/` (recursively, no manifest file to maintain), then `elaborate`s just the module you name as `TOPLEVEL` — unused modules sit unreferenced in the library. There's no `inputs/dut_src_list_*.txt` per block anymore; that whole category of file (and the staleness it kept causing every time RTL moved) is gone.

**The submodule doesn't auto-update.** If `tr-vpu_rtl`'s RTL changes, this checkout won't see it until you run:

```sh
git submodule update --remote tr-vpu_rtl
```

## Prerequisites

- Cadence Genus (confirmed working against `25.09-s002`-era tooling).
- A valid Cadence license server reachable via `CDS_LIC_FILE`.
- `tcsh` (the invocations below assume it).
- The submodule initialized (see above).

## Running synthesis

All commands run from `workspace/`. **Escaped quotes are required** for `-execute` in this environment — confirmed working:

```tcsh
cd workspace
genus -execute \"set TOPLEVEL <block_name>\" -f ../scripts/genus.tcl -log <block_name>_synth
```

`<block_name>` is any module name that exists under `tr-vpu_rtl/rtl_soc/` or `tr-vpu_rtl/rtl_baseline/` (e.g. `tr_nonlinear_vpu`, `tr_soc_top_int`, `tr_gelu`, `ibert_gelu`, ...) — no separate registration needed. This runs the full flow: reads libraries, reads and elaborates the RTL, reads `dut.sdc`, synthesizes (generic → technology-mapped → optimized) at high effort, then **exports the design** (`export/<TOPLEVEL>/post_synth/<TOPLEVEL>.{db,v,sdf,sdc}`) before running any reports — this ordering is deliberate: synthesis is the expensive multi-hour step, so the result is saved to disk immediately, before a report command that fails can cost anything more than that one report. Post-synthesis reports (`report_area`, `report_gates`, `report_hierarchy`, `report_clock_gating`, `report_design_rules`, `report_dp`, `report_qor`, `report_power`) land in `reports/<TOPLEVEL>/synthesis/post_opt/`.

### Power: vectorless

Once a design has been synthesized (its `.db` exists in `export/<TOPLEVEL>/post_synth/`), get a fast power estimate (default 12.5% toggle-rate assumption, not real activity) without re-running synthesis:

```tcsh
cd workspace
genus -execute \"set TOPLEVEL <block_name>\" -f ../scripts/report_power.tcl -log <block_name>_power
```

Report lands at `reports/<TOPLEVEL>/synthesis/post_opt/report_power.rpt`. Takes seconds, since it reads back the exported `.db` instead of re-synthesizing.

### Power: activity-based (real energy)

For power driven by actual simulation activity (a VCD from an `*_energy_tb.sv` pilot testbench on the `tr-vpu_rtl` branch) instead of a toggle-rate guess:

```tcsh
cd workspace
setenv ACTIVITY_VCD <path_to>.vcd
setenv ACTIVITY_INSTANCE <testbench_module>/<dut_instance>
genus -execute \"set TOPLEVEL <block_name>\" -f ../scripts/report_power_activity.tcl -log <block_name>_power_activity
```

Report lands at `reports/<TOPLEVEL>/synthesis/post_opt/report_power_activity.rpt`. `Energy = Total_Power(µW) × N_cycles × T_clock(ns)`, `T_clock = 3.3 ns` (`dut.defines`' `CLK_PERIOD`). All three environment variables are set individually rather than combined into one `-execute` string or multiple `-execute` flags — both of those forms broke under this environment's Genus CLI parsing; this pattern is the one confirmed to work.

**Careful:** `report_power.tcl` and `report_power_activity.tcl` both write to the same `report_power.rpt`/`report_power_activity.rpt` path per `TOPLEVEL` — running either again overwrites the previous result. Save off a report before re-running if you need to keep it.

### Fmax sweep

`run_sweep.tcl` elaborates once, snapshots the pre-synthesis design, then walks the clock period down from `START_PERIOD` to `END_PERIOD` in `STEP`-ns decrements (low synthesis effort, for speed), stopping at the first period with negative slack or a DRV violation. It then re-synthesizes the last clean period once more at **full** effort and exports that as the result — the sweep's own low-effort runs are for finding the right period fast, not for reporting.

```tcsh
cd workspace
setenv START_PERIOD 10
setenv END_PERIOD 2
setenv STEP 1
genus -execute \"set TOPLEVEL <block_name>\" -f ../scripts/run_sweep.tcl -log <block_name>_sweep
```

(`START_PERIOD`/`END_PERIOD`/`STEP` default to 10/2/1 ns if unset.) Per-period results (period, slack, DRV count) land in `reports/<TOPLEVEL>/sweep/clock_sweep_results.csv`; the final high-effort export lands at `export/<TOPLEVEL>/post_synth/<TOPLEVEL>.sweep_optimal.db`.

## Known limitations

- **`scripts/settings.tcl`, `scripts/simulation.tcl`** are currently unreferenced by anything in this flow (OCV/timing settings and a gate-level-simulation VCD-probe helper, respectively) — left over from an earlier iteration, not broken, just unused right now.
- There's no place-and-route, gate-level-simulation-with-backannotation, or Voltus power-rail flow in this branch currently — an earlier attempt at these existed as a cluster of broken, never-adapted template scripts (hardcoded to a placeholder `"average"` design) and has been removed rather than kept as dead weight. If any of that's needed later, it'll need building from scratch against this branch's actual conventions, not resurrecting the old templates.
