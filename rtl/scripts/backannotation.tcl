# This script run SDF backannotation with a GateLevel Netlist
# Specifically, the script:
#       1) Opens a waveform with all the signals of the DUT
#       2) Opens a VCD database and dumps all signals into a VCD file
#       3) Opens a TCF database and dumps all signals into a TCF file
#       4) Runs the simulation
#   execute this script with the command: 
#           simvision -input backannotation.tcl 
#       or alternatively
#           irun -input backannotation.tcl 
set design(TOPLEVEL) "average" 
set mmmc_or_simple "simple"
set debug_file "debug.xrun.txt"
set phys_synth_type "none" 
set runtype "none"

# Read in the project definitions/variables
source ../inputs/average.defines
# Open a waveform and add the relevant signals to the waveform

# Open a VCD database and start probing signals and dumping into it
database -open $design(vcd_file) -vcd -into $design(vcd_file) 
probe -database wvs -create "$design(tb_name).$design(dut_name)" -vcd -all -database $design(vcd_file) -depth 9
# The VCD probe will continue until the end of the simulation
 

