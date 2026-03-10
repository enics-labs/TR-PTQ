# ####################################################################

#  Created by Genus(TM) Synthesis Solution 25.12-s067_1 on Tue Mar 10 19:23:29 IST 2026

# ####################################################################

set sdc_version 2.0

set_units -capacitance 1000fF
set_units -time 1000ps

# Set the current design
current_design softmax_engine

set_clock_gating_check -setup 0.0 
set_wire_load_mode "top"
