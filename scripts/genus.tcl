if { [info exists env(TOPLEVEL)] } {
    set design(TOPLEVEL) $env(TOPLEVEL)
} elseif { [info exists TOPLEVEL] } {
    set design(TOPLEVEL) $TOPLEVEL
} else {
    puts "Error: TOPLEVEL not provided."
    puts "Usage (from workspace): genus -execute \"set TOPLEVEL <module>\" -f ../scripts/genus.tcl"
    exit 1
}
set debug_file "$design(TOPLEVEL).txt"
set runtype "synthesis"

# Variables
set mmmc_or_simple "simple"; # "simple" - using "read_libs"
                             # "mmmc"   - using "read_mmmc"
set phys_synth_type "none";  # "none"   - don't read any physical data
                             # "lef"    - only read lef and qrctech files
                             # "floorplan"    - read in a def of the floorplan

# Load general procedures
source ../scripts/procedures.tcl -quiet
enics_start_stage "start"

# Load the specific definitions for this project
source ../inputs/dut.defines -quiet

# Load the library paths and definitions for this technology
source ../libraries/libraries.$TECHNOLOGY.tcl -quiet
source ../libraries/libraries.$SC_TECHNOLOGY.tcl -quiet
source ../libraries/libraries.$SRAM_TECHNOLOGY.tcl -quiet
if {$design(FULLCHIP_OR_MACRO)=="FULLCHIP"} {
    source ../libraries/libraries.$IO_TECHNOLOGY.tcl -quiet
}

#############################################
#       Print values to debug file
#############################################
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

##########################
# General Genus Settings
##########################
set_db source_verbose true ; # Sourcing files will be reported as verbose
set_db information_level 9 ; # The log file will report everything
suppress_messages "PHYS-90 LBR-415"

##########################
# Read Libraries
##########################
enics_start_stage "init_design"

if {$mmmc_or_simple=="mmmc"} {
    read_mmmc $design(mmmc_view_file)
} else {
    set_db init_lib_search_path $paths(LIB_paths) 
    suppress_messages $tech(SC_LIB_SUPPRESS_MESSAGES_GENUS)
    read_libs $tech_files(ALL_WC_LIBS)
}
# Get rid of recurring unusable library cells messages, after debugging it once...
# suppress_messages "LBR-415"

##########################
# Read RTL
##########################
enics_start_stage "read_rtl"

set_db hdl_language v2001 -quiet

# Read every .sv under rtl_soc/ and rtl_baseline/ -- no per-block file
# manifest to keep in sync. elaborate (below) picks out just $TOPLEVEL;
# everything else sits unreferenced in the library.
set rtl_files [enics_glob_recursive $design(rtl_dirs) "*.sv"]
if {[llength $rtl_files] == 0} {
    puts "ENICSINFO: ERROR -- no .sv files found under $design(rtl_dirs)."
    puts "ENICSINFO: Is the RTL submodule/checkout present at the repo root?"
    exit 1
}
enics_message "Reading [llength $rtl_files] RTL files from $design(rtl_dirs)" medium
read_hdl -language sv $rtl_files

##########################
# Elaborate
##########################
enics_start_stage "elaborate"

set_db hdl_track_filename_row_col true -quiet; # helps with debug
set_db lp_insert_clock_gating true 

elaborate $design(TOPLEVEL) ;#-update

enics_start_stage "post_elaboration"

check_design -unresolved
check_design -all > $design(synthesis_reports)/post_elaboration/check_design_post_elab.rpt
if {[check_design -status]} {
    Puts "ENICSINFO: ############# There is an issue with check_design. You better look at it! ###########"
}
#save elaborated design
write_design -base_name $design(export_dir)/post_elaboartion/$design(TOPLEVEL)

# Read in a floorplan for physical synthesis
# read_def $design(floorplan_def)

##########################
# read constraints
##########################
set_db detailed_sdc_messages true ; # helps read_sdc debug
read_sdc $design(functional_sdc) -stop_on_errors 
check_timing_intent
check_timing_intent -verbose > $design(synthesis_reports)/post_elaboration/check_timing_post_elab.rpt

###################################################################################
## Define cost groups (reg2reg, in2reg, reg2out, in2out)
###################################################################################
enics_default_cost_groups
enics_report_timing $design(synthesis_reports)

################################
# clock gating settings
################################
set_db [get_db design:$design(TOPLEVEL)] .lp_clock_gating_min_flops 8
set_db [get_db design:$design(TOPLEVEL)] .lp_clock_gating_style latch 

# Prevent specific modules from being ungrouped
# set_db [get_db modules max_sub] .ungroup_ok false
# set_db [get_db modules tr_exp*] .ungroup_ok false
# set_db [get_db modules tr_reciprocal*] .ungroup_ok false
set_db auto_ungroup none

##########################
#     Synthesize
##########################
enics_start_stage "synthesis"

# Set Synthesis Efforts
set_db syn_generic_effort high
set_db syn_map_effort high
set_db syn_opt_effort high
suppress_messages "ST-110 ST-112"

if {$phys_synth_type == "floorplan"} {
    # Synthesize to generics and place generics in floorplan
    enics_start_stage "syn_generic"
    syn_generic -physical
    # Map to technology
    enics_start_stage "technology_mapping"
    syn_map -physical
    enics_report_timing $design(synthesis_reports) 
    # Post synthesis optimization
    enics_start_stage "post_syn_opt"
    syn_opt -physical
} else {
    # Synthesize to generics (non physical-aware)
    enics_start_stage "syn_generic"
    syn_generic 
    # Map to technology (non physical-aware)
    enics_start_stage "technology_mapping"
    syn_map 
    enics_report_timing $design(synthesis_reports)
    enics_start_stage "post_syn_opt"
    if {$phys_synth_type == "lef"} {
        syn_opt ;#-physical
    } else {
        syn_opt 
    }
}

#############################
#   Exporting the Design
#############################
# Moved ahead of the post-synthesis reports on purpose: syn_opt is the
# expensive, multi-hour step, and it had already finished cleanly the run
# that hit the -summary crash below -- but because no checkpoint had been
# saved yet, that completed synthesis was unrecoverable and the whole run
# had to be redone from scratch. Saving the db/netlist/sdf/sdc immediately
# once synthesis is done means a report-command bug (like that one) can
# only cost the report, never the synthesis result itself.
enics_start_stage "export_design"
write_db $design(TOPLEVEL) -to_file "$design(export_dir)/post_synth/$design(TOPLEVEL).db"
write_design -base_name "$design(export_dir)/post_synth/$design(TOPLEVEL)"
write_hdl > $design(postsyn_netlist)
write_sdf > "$design(export_dir)/post_synth/$design(TOPLEVEL).sdf"
write_sdc > "$design(export_dir)/post_synth/$design(TOPLEVEL).sdc"
# write_design -innovus -db -base_name "$design(export_dir)/pwr/genus/$design(TOPLEVEL)"

#############################
#     Post Synthesis Reports
#############################
# Each call is wrapped in catch so a single bad report (e.g. an invalid
# flag, as just happened) prints an error and moves on instead of aborting
# the whole script -- the design is already safely exported above by the
# time any of this runs, but there's no reason a typo in one report
# command should also cost the other N-1 reports.
if {[catch {enics_report_timing $design(synthesis_reports)} err]} {
    enics_message "enics_report_timing failed: $err" medium
}
if {[catch {enics_report_timing_full $design(synthesis_reports) 50} err]} {
    enics_message "enics_report_timing_full failed: $err" medium
}
set post_synth_reports [list \
    report_area \
    report_gates \
    report_hierarchy \
    report_clock_gating \
    report_design_rules \
    report_dp \
    report_qor \
    report_power \
]
foreach rpt $post_synth_reports {
    enics_message "$rpt" medium
    if {[catch {
        $rpt
        $rpt > "$design(synthesis_reports)/post_opt/${rpt}.rpt"
    } err]} {
        enics_message "$rpt failed: $err" medium
    }
}
