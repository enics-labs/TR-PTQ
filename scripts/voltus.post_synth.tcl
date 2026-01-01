puts "*****************************************************"
puts "***     MAKE SURE YOU RAN voltus -stylus !!!!!    ***"
puts "*****************************************************"

set design(TOPLEVEL) "average"
set genus_view "default_emulate_view"
set debug_file "debug.voltus.txt"
set mmmc_or_simple "simple"
set debug_file "debug.xrun.txt"
set phys_synth_type "lef"
set runtype "none"

# Load general procedures
source ../scripts/procedures.tcl -quiet
# Load the specific definitions for this project
source ../inputs/$design(TOPLEVEL).defines -quiet
# Load general settings
source ../scripts/settings.tcl -quiet

# Load the library paths and definitions for this technology
source ../libraries/libraries.$TECHNOLOGY.tcl -quiet
source ../libraries/libraries.$SC_TECHNOLOGY.tcl -quiet

# Restore design
#source $design(export_dir)/pwr/genus/$design(TOPLEVEL).invs_init.tcl
read_db $design(export_dir)/pwr/genus/$design(TOPLEVEL).db

set_db power_method static
set_db power_report_missing_nets true
#set_db power_view $genus_view
set_analysis_view -setup $genus_view -hold $genus_view \
            -leakage $genus_view -dynamic $genus_view


# TCF
#read_activity_file -reset 
#read_activity_file -format TCF -scope $design(dut_name) $design(tcf_file)

# VCD
read_activity_file -reset 
read_activity_file -format VCD -scope $design(tb_name).$design(dut_name) \
    -start $design(power_report_start_time) -end $design(power_report_end_time) \
    $design(export_dir)/pwr/$design(TOPLEVEL).vcd

report_power -rail_analysis_format VS -out_file $design(export_dir)/pwr/power.rpt
