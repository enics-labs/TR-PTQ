# report_power.tcl -- power report for an already-synthesized design.
# Reads back the post_synth .db a prior genus.tcl run already exported
# (write_db in genus.tcl) instead of re-running read_hdl/elaborate/
# syn_generic/syn_map/syn_opt, so this takes seconds, not hours.
#
# Usage (from workspace): genus -execute "set TOPLEVEL <module>" -f ../scripts/report_power.tcl

if { [info exists env(TOPLEVEL)] } {
    set design(TOPLEVEL) $env(TOPLEVEL)
} elseif { [info exists TOPLEVEL] } {
    set design(TOPLEVEL) $TOPLEVEL
} else {
    puts "Error: TOPLEVEL not provided."
    puts {Usage (from workspace): genus -execute "set TOPLEVEL <module>" -f ../scripts/report_power.tcl}
    exit 1
}
set debug_file "$design(TOPLEVEL).power.txt"

# dut.defines' own trailing debug-dump loop logs these three unconditionally
# (it expects genus.tcl to have set them already, at genus.tcl's own top);
# this script skips the full flow those belong to, so set them here instead.
set runtype "synthesis"
set mmmc_or_simple "simple"
set phys_synth_type "none"

source ../scripts/procedures.tcl -quiet
source ../inputs/dut.defines -quiet
source ../libraries/libraries.$TECHNOLOGY.tcl -quiet
source ../libraries/libraries.$SC_TECHNOLOGY.tcl -quiet
source ../libraries/libraries.$SRAM_TECHNOLOGY.tcl -quiet
if {$design(FULLCHIP_OR_MACRO)=="FULLCHIP"} {
    source ../libraries/libraries.$IO_TECHNOLOGY.tcl -quiet
}

set_db information_level 9
suppress_messages "PHYS-90 LBR-415"
set_db init_lib_search_path $paths(LIB_paths)
suppress_messages $tech(SC_LIB_SUPPRESS_MESSAGES_GENUS)
read_libs $tech_files(ALL_WC_LIBS)

set pwr_db "$design(export_dir)/post_synth/$design(TOPLEVEL).db"
if {![file exists $pwr_db]} {
    puts "ENICSINFO: ERROR -- $pwr_db not found. Has $design(TOPLEVEL) been synthesized yet?"
    exit 1
}
read_db $pwr_db

file mkdir "$design(synthesis_reports)/post_opt"
report_power           > "$design(synthesis_reports)/post_opt/report_power.rpt"
report_power -hierarchy > "$design(synthesis_reports)/post_opt/report_power_hierarchy.rpt"

puts "ENICSINFO: power report written for $design(TOPLEVEL) -> $design(synthesis_reports)/post_opt/report_power.rpt"
exit
