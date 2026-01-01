set design(TOPLEVEL) "vec_mac_requant"
set debug_file "debug.txt"
set runtype "synthesis"

# Variables
set mmmc_or_simple "simple"; # "simple" - using "read_libs"
                             # "mmmc"   - using "read_mmmc"
set phys_synth_type "lef" ;  # "none"   - don't read any physical data
                             # "lef"    - only read lef and qrctech files
                             # "floorplan"    - read in a def of the floorplan


# Load general procedures
source ../scripts/procedures.tcl -quiet
enics_start_stage "start"


# Load the specific definitions for this project
source ../inputs/$design(TOPLEVEL).defines -quiet

# Load the library paths and definitions for this technology
source ../inputs/libraries.$TECHNOLOGY.tcl -quiet
source ../inputs/libraries.$SC_TECHNOLOGY.tcl -quiet
source ../inputs/libraries.$SRAM_TECHNOLOGY.tcl -quiet
if {$design(FULLCHIP_OR_MACRO)=="FULLCHIP"} {
    source ../inputs/libraries.$IO_TECHNOLOGY.tcl -quiet
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
suppress_messages "PHYS-90"

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
suppress_messages "LBR-415"

############################
# Physical Synthesis Setup
############################
if {$phys_synth_type == "none"} {
   enics_message "Physical Synthesis is disabled"
   read_qrc $tech_files(QRCTECH_FILE_WC)
} else {
    suppress_messages $tech(SC_LEF_SUPPRESS_MESSAGES_GENUS)
    read_physical -lef $tech_files(ALL_LEFS)
    if {$phys_synth_type == "floorplan"} { 
        # You need to read a .def file for the floorplan to enable physical synthesis 
        read_def $design(floorplan_def)
    }
}

##########################
# Read RTL
##########################
enics_start_stage "read_rtl"

set_db init_hdl_search_path $design(hdl_search_paths)
set_db hdl_language v2001 -quiet

# Stop reporting elaboration of multi-dimensional arrays
suppress_messages "CDFG-250"
# Stop reporting implementation inference (chipWare)
suppress_messages "CWD-19 CWD-36"
# Stop reporting Replaced logic with a constant value
suppress_messages "CDFG-771"
read_hdl -language sv -f $design(read_hdl_list)

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

##########################
#     Synthesize
##########################
enics_start_stage "synthesis"

# Set Synthesis Efforts
set_db syn_generic_effort low
set_db syn_map_effort low
set_db syn_opt_effort low
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
#     Post Synthesis Reports
#############################
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

#############################
#   Exporting the Design
#############################
enics_start_stage "export_design"
write_db $design(TOPLEVEL) -to_file "$design(export_dir)/post_synth/$design(TOPLEVEL).db" 
write_design -base_name "$design(export_dir)/post_synth/$design(TOPLEVEL)" -innovus
write_hdl > $design(postsyn_netlist)
write_sdf > "$design(export_dir)/post_synth/$design(TOPLEVEL).sdf"
write_sdc > "$design(export_dir)/post_synth/$design(TOPLEVEL).sdc" 
write_design -innovus -db -base_name "$design(export_dir)/pwr/genus/$design(TOPLEVEL)"
