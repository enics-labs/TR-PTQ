#!/bin/bash
# Unattended synthesis + activity-based power of the operator comparison (dedicated tr_* / ibert_* modules and the shared tr_nonlinear_vpu).
#
# Usage (detached, survives logout; resumable: finished steps are skipped):
#   source /tools/ocs/default/common/settings.sh
#   /tools/common/wrappers/qrun -n 1 -m 12G bash scripts/run_comparison.sh
# Progress: ../TR-PTQ_runs/driver.log      Results: reports_cmp/ (committed and pushed at the end), summary in reports_cmp/summary.md
#
# Each configuration runs in its own directory ../TR-PTQ_runs/<tag>/ (own workspace, reports and export; scripts, inputs, libraries and the RTL are
# symlinks to this clone), so nothing in reports/ is touched. Environment hooks used by the flow: IO_DELAY_FRAC (inputs/dut.defines), CLOCK_GATING
# (scripts/genus.tcl). Edit CONFIGS / MODULES below to change the matrix.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNS=$(cd "$ROOT/.." && pwd)/TR-PTQ_runs
PNX=${PNX:-/project/generic/users/yonatap/ws/ddp23_pulpenix/ddp23_pnx}
VCD_DIR=$PNX/src/ips/tr_ptq/workspace                      # VCDs of the *_energy_tb pilots (tr-vpu_rtl workspace/)
GENUS=/tools/cadence/DDI/25.12/GENUS251/bin/genus           # the real Genus (the PATH "genus" is a queueing wrapper)
export CDS_LIC_FILE=5280@enicsw01
LOG=$RUNS/driver.log
mkdir -p "$RUNS"

ALL="tr_rmsnorm ibert_rmsnorm tr_swiglu ibert_swiglu tr_softmax ibert_softmax tr_nonlinear_vpu"
# tag | clock gating | I/O delay fraction | modules
CONFIGS=(
  "gated_io10|true|0.1|$ALL"
  "nogate_io10|false|0.1|$ALL"
  "gated_io30|true|0.3|tr_nonlinear_vpu"
)

# test hooks: TEST_CONFIG="tag|clock_gating|io_frac|modules" replaces the matrix; NO_PUBLISH=1 stops after the runs
[ -n "${TEST_CONFIG:-}" ] && CONFIGS=("$TEST_CONFIG")

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# power-activity inputs of a module: "vcd|instance|suffix" lines
vcd_list() {
  local m=$1
  if [ "$m" = tr_nonlinear_vpu ]; then
    for op in rmsnorm swiglu softmax; do
      echo "$VCD_DIR/tr_nonlinear_vpu_${op}_power_activity.vcd|tr_nonlinear_vpu_${op}_energy_tb/dut|_$op"
    done
  else
    echo "$VCD_DIR/${m}_power_activity.vcd|${m}_energy_tb/dut|"
  fi
}

run_genus() {   # run_genus <rundir> <tcl> <logname> [VAR=value ...]
  local d=$1 tcl=$2 lg=$3; shift 3
  ( cd "$d/workspace" && env "$@" "$GENUS" -batch -no_gui -files "../scripts/$tcl" -log "$lg" > "$lg.stdout" 2>&1 )
}

setup_rundir() {
  local d=$RUNS/$1
  mkdir -p "$d/workspace" "$d/reports" "$d/export"
  for x in scripts inputs libraries tr-vpu_rtl dbs; do
    [ -e "$ROOT/$x" ] && ln -sfn "$ROOT/$x" "$d/$x"
  done
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r tag cg io mods <<< "$cfg"
  setup_rundir "$tag"
  D=$RUNS/$tag
  log "=== config $tag  CLOCK_GATING=$cg IO_DELAY_FRAC=$io  modules: $mods"
  for m in $mods; do
    P=$D/reports/$m/synthesis/post_opt
    if [ -f "$P/report_qor.rpt" ]; then
      log "[$tag/$m] synthesis already done"
    else
      ok=0
      for attempt in 1 2; do
        log "[$tag/$m] synthesis (attempt $attempt)"
        t0=$(date +%s)
        run_genus "$D" genus.tcl "${m}_synth" TOPLEVEL="$m" CLOCK_GATING="$cg" IO_DELAY_FRAC="$io"
        if [ -f "$P/report_qor.rpt" ] && [ -f "$D/export/$m/post_synth/$m.db" ]; then ok=1; break; fi
        log "[$tag/$m] synthesis attempt $attempt FAILED (see $D/workspace/${m}_synth.log)"
      done
      [ $ok = 1 ] && log "[$tag/$m] synthesis OK ($(( $(date +%s) - t0 )) s)" || { log "[$tag/$m] GIVING UP on synthesis"; continue; }
    fi
    vcd_list "$m" | while IFS='|' read -r vcd inst sfx; do
      out=$P/report_power_activity${sfx}.rpt
      if [ -f "$out" ]; then log "[$tag/$m] power${sfx} already done"; continue; fi
      if [ ! -f "$vcd" ]; then log "[$tag/$m] MISSING VCD $vcd"; continue; fi
      log "[$tag/$m] activity power${sfx} ($inst)"
      rm -f "$P/report_power_activity.rpt"
      run_genus "$D" report_power_activity.tcl "${m}_power_activity${sfx}" TOPLEVEL="$m" CLOCK_GATING="$cg" IO_DELAY_FRAC="$io" ACTIVITY_VCD="$vcd" ACTIVITY_INSTANCE="$inst"
      if [ -f "$P/report_power_activity.rpt" ]; then
        [ -n "$sfx" ] && cp "$P/report_power_activity.rpt" "$out"
        log "[$tag/$m] power${sfx} OK"
      else
        log "[$tag/$m] power${sfx} FAILED (see $D/workspace/${m}_power_activity${sfx}.log)"
      fi
    done
  done
done

log "=== all runs finished"
[ -n "${NO_PUBLISH:-}" ] && { log "NO_PUBLISH set, stopping"; exit 0; }
log "summarizing"
cd "$ROOT" || exit 1
python3 scripts/summarize_comparison.py --runs "$RUNS" --root "$ROOT" --out "$ROOT/reports_cmp" >> "$LOG" 2>&1 || log "summary FAILED"

# ---- publish: synthesis branch (this clone), then the integration repository ----
publish() {   # publish <repo> <branch> <message> <paths...>
  local repo=$1 br=$2 msg=$3; shift 3
  ( cd "$repo" || exit 1
    git add "$@" || exit 1
    if git diff --cached --quiet; then echo "nothing to commit in $repo"; exit 0; fi
    git commit -q -m "$msg

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" || exit 1
    git push origin "$br" || { git pull --rebase --autostash origin "$br" && git push origin "$br"; }
  ) >> "$LOG" 2>&1 && log "published $repo ($br)" || log "PUBLISH FAILED for $repo"
}
publish "$ROOT" tr-vpu_synthesis "Operator comparison runs: clock-gated VPU, 10% boundary delay, gating-off diagnostic

Runs of the three dedicated Taylor-Region modules, the three divider-based modules and tr_nonlinear_vpu (reports_cmp/, summary in reports_cmp/summary.md).
Flow hooks: IO_DELAY_FRAC in inputs/dut.defines, CLOCK_GATING in scripts/genus.tcl; driver scripts/run_comparison.sh." \
  reports_cmp scripts/run_comparison.sh scripts/summarize_comparison.py scripts/genus.tcl inputs/dut.defines README.md tr-vpu_rtl
if [ -f "$ROOT/reports_cmp/summary.md" ] && [ -d "$PNX/.git" ]; then
  cp "$ROOT/reports_cmp/summary.md" "$PNX/docs/llama/standalone-comparison-results.md"
  publish "$PNX" llama-research "Standalone operator comparison: generated results (clock-gated VPU, 10% boundary delay, gating-off diagnostic)

Generated by scripts/run_comparison.sh and summarize_comparison.py in the synthesis branch (tr-vpu_synthesis)." \
    docs/llama/standalone-comparison-results.md src/ips/tr_ptq
fi
log "=== done"
