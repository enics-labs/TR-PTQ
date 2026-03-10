# Cadence Genus(TM) Synthesis Solution, Version 25.12-s067_1, built Nov 17 2025 13:10:10

# Date: Tue Mar 10 19:09:29 2026
# Host: enicsw14.local (x86_64 w/Linux 5.14.0-570.17.1.el9_6.x86_64) (12cores*48cpus*2physical cpus*Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz 30720KB)
# OS:   Rocky Linux 9.6 (Blue Onyx)

set design(TOPLEVEL) "softmax_engine"
set debug_file "debug_softmax.txt"
set runtype "synthesis"
set mmmc_or_simple "simple"; # "simple" - using "read_libs"
set phys_synth_type "none";  # "none"   - don't read any physical data
source ../scripts/procedures.tcl -quiet
enics_start_stage "start"
source ../inputs/$design(TOPLEVEL).defines -quiet
source ../inputs/libraries.$TECHNOLOGY.tcl -quiet
source ../inputs/libraries.$SC_TECHNOLOGY.tcl -quiet
source ../inputs/libraries.$SRAM_TECHNOLOGY.tcl -quiet
if {$design(FULLCHIP_OR_MACRO)=="FULLCHIP"} {
    source ../inputs/libraries.$IO_TECHNOLOGY.tcl -quiet
}
source ../libraries/libraries.$TECHNOLOGY.tcl -quiet
source ../libraries/libraries.$SC_TECHNOLOGY.tcl -quiet
source ../libraries/libraries.$SRAM_TECHNOLOGY.tcl -quiet
if {$design(FULLCHIP_OR_MACRO)=="FULLCHIP"} {
    source ../libraries/libraries.$IO_TECHNOLOGY.tcl -quiet
}
set df [open $debug_file a]
puts $df "\n******************************************"
puts $df "* Debug values after everything was loaded *"
puts $df "******************************************"
foreach dic {paths tech tech_files design} {
    foreach key [array names $dic] {
        puts $df "${dic}(${key}) = \t[set ${dic}([set key])]"
    }
}
close $df
set_db source_verbose true ; # Sourcing files will be reported as verbose
set_db information_level 9 ; # The log file will report everything
suppress_messages "PHYS-90 LBR-415"
enics_start_stage "init_design"
set_db init_lib_search_path $paths(LIB_paths)
suppress_messages $tech(SC_LIB_SUPPRESS_MESSAGES_GENUS)
read_libs $tech_files(ALL_WC_LIBS)
enics_start_stage "read_rtl"
set_db init_hdl_search_path $design(hdl_search_paths)
set_db hdl_language v2001 -quiet
read_hdl -language sv -f $design(read_hdl_list)
enics_start_stage "elaborate"
set_db hdl_track_filename_row_col true -quiet; # helps with debug
set_db lp_insert_clock_gating true
elaborate $design(TOPLEVEL)
enics_start_stage "post_elaboration"
check_design -unresolved
check_design -all > $design(synthesis_reports)/post_elaboration/check_design_post_elab.rpt
if {[check_design -status]} {
    Puts "ENICSINFO: ############# There is an issue with check_design. You better look at it! ###########"
}
write_design -base_name $design(export_dir)/post_elaboartion/$design(TOPLEVEL)
set_db detailed_sdc_messages true ; # helps read_sdc debug
read_sdc $design(functional_sdc) -stop_on_errors
check_timing_intent
check_timing_intent -verbose > $design(synthesis_reports)/post_elaboration/check_timing_post_elab.rpt
enics_default_cost_groups
enics_report_timing $design(synthesis_reports)set_db [get_db design:$design(TOPLEVEL)] .lp_clock_gating_min_flops 8
set_db [get_db design:$design(TOPLEVEL)] .lp_clock_gating_style latch
enics_report_timing $design(synthesis_reports)
set_db [get_db design:$design(TOPLEVEL)] .lp_clock_gating_min_flops 8
set_db [get_db design:$design(TOPLEVEL)] .lp_clock_gating_style latch
enics_start_stage "synthesis"
set_db syn_generic_effort low
set_db syn_map_effort low
set_db syn_opt_effort low
suppress_messages "ST-110 ST-112"
enics_start_stage "syn_generic"
syn_generic
enics_start_stage "technology_mapping"
syn_map
enics_report_timing $design(synthesis_reports)
enics_start_stage "post_syn_opt"
syn_opt
enics_report_timing $design(synthesis_reports)
set post_synth_reports [list \
    report_area \
    report_gates \
    report_hierarchy \
    report_clock_gating \
    report_design_rules \
    report_dp \
    report_qor \
]
foreach rpt $post_synth_reports {
    enics_message "$rpt" medium
    $rpt
    $rpt > "$design(synthesis_reports)/post_opt/${rpt}.rpt"
}
enics_start_stage "export_design"
write_db $design(TOPLEVEL) -to_file "$design(export_dir)/post_synth/$design(TOPLEVEL).db"
write_design -base_name "$design(export_dir)/post_synth/$design(TOPLEVEL)" -innovus
write_hdl > $design(postsyn_netlist)
write_sdf > "$design(export_dir)/post_synth/$design(TOPLEVEL).sdf"
write_sdc > "$design(export_dir)/post_synth/$design(TOPLEVEL).sdc"
write_design -innovus -db -base_name "$design(export_dir)/pwr/genus/$design(TOPLEVEL)"
enics_start_stage "export_design"
write_db $design(TOPLEVEL) -to_file "$design(export_dir)/post_synth/$design(TOPLEVEL).db"
write_design -base_name "$design(export_dir)/post_synth/$design(TOPLEVEL)"
write_hdl > $design(postsyn_netlist)
write_sdf > "$design(export_dir)/post_synth/$design(TOPLEVEL).sdf"
write_sdc > "$design(export_dir)/post_synth/$design(TOPLEVEL).sdc"
write_design -innovus -db -base_name "$design(export_dir)/pwr/genus/$design(TOPLEVEL)"
enics_start_stage "export_design"
write_db $design(TOPLEVEL) -to_file "$design(export_dir)/post_synth/$design(TOPLEVEL).db"
write_design -base_name "$design(export_dir)/post_synth/$design(TOPLEVEL)"
write_hdl > $design(postsyn_netlist)
git status
