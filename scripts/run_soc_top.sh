#!/bin/bash
# Top-level synthesis of tr_soc_top_int in this clone's workspace/ (default flow: clock gating inferred, 30% boundary delay).
# Usage: source /tools/ocs/default/common/settings.sh; /tools/common/wrappers/qrun -n 1 -m 16G bash scripts/run_soc_top.sh
export CDS_LIC_FILE=5280@enicsw01
cd "$(dirname "$0")/../workspace" || exit 1
unset TOPLEVEL
rm -f tr_soc_top_int_synth.done
env TOPLEVEL=tr_soc_top_int /tools/cadence/DDI/25.12/GENUS251/bin/genus -batch -no_gui -files ../scripts/genus.tcl -log tr_soc_top_int_synth > tr_soc_top_int_synth.stdout 2>&1
echo "genus rc=$?" > tr_soc_top_int_synth.done
