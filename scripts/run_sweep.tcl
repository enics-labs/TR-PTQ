# run_sweep.tcl -- Fmax sweep: starting from a relaxed clock period, walks
# down in $STEP-ns decrements (low synthesis effort, for exploration
# speed) until setup slack goes negative or a DRV violation appears, then
# re-synthesizes once more at the last clean period at FULL (high) effort
# and exports that as the result -- the sweep loop's own low-effort runs
# are for finding the right period fast, not for reporting/handoff.
#
# Usage (from workspace): genus -execute "set TOPLEVEL <module>" -f ../scripts/run_sweep.tcl
# Optional range override (ns, all default if unset): setenv START_PERIOD/END_PERIOD/STEP

if { [info exists env(TOPLEVEL)] } {
    set design(TOPLEVEL) $env(TOPLEVEL)
} elseif { [info exists TOPLEVEL] } {
    set design(TOPLEVEL) $TOPLEVEL
} else {
    puts "Error: TOPLEVEL not provided."
    puts {Usage (from workspace): genus -execute "set TOPLEVEL <module>" -f ../scripts/run_sweep.tcl}
    exit 1
}
set debug_file "$design(TOPLEVEL).sweep.txt"
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

##########################
# Read RTL + Elaborate (same wholesale read as genus.tcl -- see
# enics_glob_recursive in procedures.tcl)
##########################
set_db hdl_language v2001 -quiet
set rtl_files [enics_glob_recursive $design(rtl_dirs) "*.sv"]
if {[llength $rtl_files] == 0} {
    puts "ENICSINFO: ERROR -- no .sv files found under $design(rtl_dirs)."
    puts "ENICSINFO: Is the RTL submodule/checkout present at the repo root?"
    exit 1
}
enics_message "Reading [llength $rtl_files] RTL files from $design(rtl_dirs)" medium
read_hdl -language sv $rtl_files
elaborate $design(TOPLEVEL)

##########################
# Snapshot the elaborated (pre-synthesis, generic) design. Each sweep
# iteration restores from here rather than re-elaborating from scratch, so
# a failed/aborted iteration can never corrupt the next one's starting
# point, and re-elaboration (the slow part) only happens once.
##########################
file mkdir "$design(reports_dir)/sweep"
set snapshot_path "$design(reports_dir)/sweep/elaborated_snapshot.db"
write_db $design(TOPLEVEL) -to_file $snapshot_path
enics_message "Snapshot written to $snapshot_path" medium

##############################################################
# SWEEP LOOP (quiet, low effort -- exploration speed over result quality)
##############################################################
set_db source_verbose false
set_db information_level 0

if {[info exists env(START_PERIOD)]} { set start_period $env(START_PERIOD) } else { set start_period 10 }
if {[info exists env(END_PERIOD)]}   { set end_period   $env(END_PERIOD)   } else { set end_period   2  }
if {[info exists env(STEP)]}         { set step         $env(STEP)         } else { set step         1  }
set last_passing_period ""
set found_clean_run false

set csv_path "$design(reports_dir)/sweep/clock_sweep_results.csv"
set csv_file [open $csv_path w]
puts $csv_file "Period_ns,Slack_ns,DRV_Violations"

for {set clk_p $start_period} {$clk_p >= $end_period} {set clk_p [expr {$clk_p - $step}]} {

    puts "\n=============================================="
    puts "  Starting Iteration with Clock Period: $clk_p ns"
    puts "=============================================="

    # A. Clean up any design left over from the previous iteration
    if {[get_db designs] != ""} {
        redirect /dev/null { delete_obj [get_db designs] }
    }
    # B. Restore the clean pre-synthesis snapshot
    redirect /dev/null { read_db $snapshot_path }
    current_design $design(TOPLEVEL)

    # C. Update timing variables for this period (same formulas as
    # dut.defines, so a sweep iteration's constraints match a real run's)
    set design(CLK_PERIOD)        $clk_p
    set design(clock_period_list) [list $clk_p]
    set design(INPUT_DELAY)       [expr {$clk_p * 0.3}]
    set design(OUTPUT_DELAY)      [expr {$clk_p * 0.3}]
    set design(INPUT_TRANSITION)  [expr {$clk_p / 10.0}]
    source $design(functional_sdc)

    # D. Synthesize at low effort
    enics_start_stage "synthesis"
    set_db syn_generic_effort low
    set_db syn_map_effort low
    set_db syn_opt_effort low
    suppress_messages "ST-110 ST-112"
    redirect /dev/null {
        foreach cell [get_db lib_cells -if {.scan_enable_pins!=""}] {set_db $cell .avoid true}
        enics_start_stage "syn_generic"
        syn_generic
        enics_start_stage "technology_mapping"
        syn_map
        enics_start_stage "post_syn_opt"
        syn_opt
    }

    # E. Check timing (worst slack)
    set worst_paths [report_timing -max_paths 1 -nworst 1 -collection]
    if {$worst_paths != ""} {
        set worst_path [lindex $worst_paths 0]
        set slack [get_db $worst_path .slack]
    } else {
        set slack 0.0
    }

    # F. Check DRVs (max transition / capacitance / fanout)
    set drv_count 0
    set rpt_file "$design(reports_dir)/sweep/temp_drv_check.rpt"
    redirect $rpt_file {
        report_constraint -drv_violation_type max_transition -all_violators
        report_constraint -drv_violation_type max_capacitance -all_violators
        report_constraint -drv_violation_type max_fanout -all_violators
    }
    if {[file exists $rpt_file]} {
        set fp [open $rpt_file r]
        set file_data [read $fp]
        close $fp
        set matches [regexp -all -inline {violation total\s*=\s*(\d+)} $file_data]
        foreach {whole_match count} $matches { set drv_count [expr {$drv_count + $count}] }
        file delete $rpt_file
    }

    puts "  -> Result at ${clk_p}ns: Slack = $slack | DRV Violations = $drv_count"
    puts $csv_file "$clk_p,$slack,$drv_count"

    # G. Decide: stop on first failure, otherwise record and continue
    if {$slack < 0.0} {
        puts "\n  FAILURE: Negative slack ($slack) at ${clk_p}ns. Stopping sweep."
        break
    } elseif {$drv_count > 0} {
        puts "\n  FAILURE: $drv_count DRV violation(s) at ${clk_p}ns (slack was $slack). Stopping sweep."
        break
    } else {
        puts "  >> SUCCESS: clean run."
        set last_passing_period $clk_p
        set found_clean_run true
    }
}
close $csv_file
enics_message "Sweep results written to $csv_path" medium

##############################################################
# FINAL RUN: re-synthesize at the best passing period, at full (high)
# effort, and export properly.
##############################################################
if {!$found_clean_run} {
    puts "\nENICSINFO: No passing period found in \[$end_period, $start_period\] ns. Nothing to finalize."
    exit 1
}

puts "\n=============================================="
puts "  Running FINAL high-effort synthesis at: $last_passing_period ns"
puts "=============================================="

if {[get_db designs] != ""} {
    redirect /dev/null { delete_obj [get_db designs] }
}
redirect /dev/null { read_db $snapshot_path }
current_design $design(TOPLEVEL)

set design(CLK_PERIOD)        $last_passing_period
set design(clock_period_list) [list $last_passing_period]
set design(INPUT_DELAY)       [expr {$last_passing_period * 0.3}]
set design(OUTPUT_DELAY)      [expr {$last_passing_period * 0.3}]
set design(INPUT_TRANSITION)  [expr {$last_passing_period / 10.0}]
source $design(functional_sdc)

set_db source_verbose true
set_db information_level 1
enics_start_stage "synthesis"
set_db syn_generic_effort high
set_db syn_map_effort high
set_db syn_opt_effort high
suppress_messages "ST-110 ST-112"

foreach cell [get_db lib_cells -if {.scan_enable_pins!=""}] {set_db $cell .avoid true}
enics_start_stage "syn_generic"
syn_generic
enics_start_stage "technology_mapping"
syn_map
enics_report_timing $design(synthesis_reports)
enics_start_stage "post_syn_opt"
syn_opt

file mkdir "$design(export_dir)/post_synth"
set sweep_db "$design(export_dir)/post_synth/$design(TOPLEVEL).sweep_optimal.db"
write_db $design(TOPLEVEL) -to_file $sweep_db

puts "\nENICSINFO: Final synthesis complete at ${last_passing_period}ns."
puts "ENICSINFO: Exported to $sweep_db"
puts "ENICSINFO: Per-period results: $csv_path"
