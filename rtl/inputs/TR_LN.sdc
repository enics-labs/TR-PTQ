

################################
#     IO Constraints
################################
set_max_delay [expr $design(CLK_PERIOD)] \
    -from [all_inputs] \
    -to   [all_outputs]


set tech(SDC_LOAD_VALUE) [lindex [get_db [get_lib_pins $tech(SDC_LOAD_PIN)] .capacitance] 0]
set_load             $tech(SDC_LOAD_VALUE)             [all_outputs]
set_input_transition $design(INPUT_TRANSITION)         [all_inputs]
set_driving_cell     -lib_cell $tech(SDC_DRIVING_CELL) [all_inputs]; # gets a string, not a collection!

################################
#     DRV Constraints
################################
set_max_fanout $design(MAX_FANOUT)  	        [current_design]
set_max_transition $design(MAX_TRANSITION) 	[current_design]
#set_max_capacitance $design(MAX_CAPACITANCE) 	[current_design]
#set_max_transition  $clk_leaf_slew -clock_path [all_clocks]
#set_max_capacitance $clk_cap -clock_path       [all_clocks]
