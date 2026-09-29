# report_power_activity.tcl -- power report driven by REAL simulation
# activity (a VCD from the *_energy_tb.sv pilot testbenches) instead of
# vectorless estimation. Reads back the already-exported post_synth .db
# (same as report_power.tcl) and layers a read_activity_file on top, so
# Dynamic Power reflects what actually switched during one real operation
# instead of a default 12.5% toggle-rate guess.
#
# Usage (from workspace, tcsh):
#   setenv ACTIVITY_VCD power_activity.vcd
#   setenv ACTIVITY_INSTANCE ibert_gelu_energy_tb/dut
#   genus -execute "set TOPLEVEL <module>" -f ../scripts/report_power_activity.tcl
#
# (all three set as plain env vars so this matches the exact single
# -execute/one-variable invocation pattern already proven to work, rather
# than relying on multiple -execute flags or semicolon-chained values in
# one -execute string, both of which Genus's own CLI parser rejected here.)

if { [info exists env(TOPLEVEL)] } {
    set design(TOPLEVEL) $env(TOPLEVEL)
} elseif { [info exists TOPLEVEL] } {
    set design(TOPLEVEL) $TOPLEVEL
} else {
    puts "Error: TOPLEVEL not provided."
    exit 1
}
if { [info exists env(ACTIVITY_VCD)] } {
    set activity_vcd $env(ACTIVITY_VCD)
} elseif { ![info exists activity_vcd] } {
    puts "Error: activity_vcd not provided. setenv ACTIVITY_VCD <path-to.vcd> before running."
    exit 1
}
if { [info exists env(ACTIVITY_INSTANCE)] } {
    set activity_instance $env(ACTIVITY_INSTANCE)
} elseif { ![info exists activity_instance] } {
    puts "Error: activity_instance not provided. setenv ACTIVITY_INSTANCE <tb_top/dut_inst> before running."
    exit 1
}

set debug_file "$design(TOPLEVEL).power_activity.txt"
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

if {![file exists $activity_vcd]} {
    puts "ENICSINFO: ERROR -- $activity_vcd not found."
    exit 1
}

# Confirmed real usage: read_vcd [-vcd_scope <string>] ... <string>
# -- the VCD path is a trailing positional argument, not a -file flag,
# and the scope flag is -vcd_scope, not -instance.
puts "ENICSINFO: loading activity from $activity_vcd (scope $activity_instance)..."
read_vcd -vcd_scope $activity_instance $activity_vcd

file mkdir "$design(synthesis_reports)/post_opt"
report_power > "$design(synthesis_reports)/post_opt/report_power_activity.rpt"
# report_power -hierarchy errors ("missing an argument for option
# -hierarchy") -- it wants a value, not a bare flag, and the exact syntax
# isn't confirmed. Dropped rather than guess a fourth time: not needed for
# the energy-per-operator number, which only uses the summary above.

puts "ENICSINFO: activity-based power report written for $design(TOPLEVEL) -> $design(synthesis_reports)/post_opt/report_power_activity.rpt"
exit
