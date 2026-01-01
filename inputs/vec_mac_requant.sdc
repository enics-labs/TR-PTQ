
# Create Clocks
foreach cname $design(clock_list) cport $design(clock_port_list) cperiod $design(CLK_PERIOD) {
    create_clock -period $cperiod -name $cname 	[get_ports $cport]
    set_clock_uncertainty $design(CLOCK_UNCERTAINTY) $cname
}

if {$runtype=="synthesis"} {
    set_ideal_network [get_ports $design(clock_port_list)]
    set_ideal_network [get_ports $design(RST_PORT)]
}
################################
#     IO Constraints
################################
set_input_delay -clock $design(CLK_NAME) $design(INPUT_DELAY) \
        [remove_from_collection [all_inputs] $design(CLK_PORT)]
set_output_delay -clock $design(CLK_NAME) $design(OUTPUT_DELAY) [all_outputs]
set_max_delay [expr $design(CLK_PERIOD)/2 + $design(INPUT_DELAY) + $design(OUTPUT_DELAY)] \
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

