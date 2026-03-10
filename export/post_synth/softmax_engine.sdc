# ####################################################################

#  Created by Genus(TM) Synthesis Solution 25.12-s067_1 on Tue Mar 10 21:31:25 IST 2026

# ####################################################################

set sdc_version 2.0

set_units -capacitance 1000fF
set_units -time 1000ps

# Set the current design
current_design softmax_engine

create_clock -name "clk" -period 10.0 -waveform {0.0 5.0} [get_ports clk]
set_load -pin_load 0.0025 [get_ports valid_out]
set_load -pin_load 0.0025 [get_ports {prob_out[7][7]}]
set_load -pin_load 0.0025 [get_ports {prob_out[7][6]}]
set_load -pin_load 0.0025 [get_ports {prob_out[7][5]}]
set_load -pin_load 0.0025 [get_ports {prob_out[7][4]}]
set_load -pin_load 0.0025 [get_ports {prob_out[7][3]}]
set_load -pin_load 0.0025 [get_ports {prob_out[7][2]}]
set_load -pin_load 0.0025 [get_ports {prob_out[7][1]}]
set_load -pin_load 0.0025 [get_ports {prob_out[7][0]}]
set_load -pin_load 0.0025 [get_ports {prob_out[6][7]}]
set_load -pin_load 0.0025 [get_ports {prob_out[6][6]}]
set_load -pin_load 0.0025 [get_ports {prob_out[6][5]}]
set_load -pin_load 0.0025 [get_ports {prob_out[6][4]}]
set_load -pin_load 0.0025 [get_ports {prob_out[6][3]}]
set_load -pin_load 0.0025 [get_ports {prob_out[6][2]}]
set_load -pin_load 0.0025 [get_ports {prob_out[6][1]}]
set_load -pin_load 0.0025 [get_ports {prob_out[6][0]}]
set_load -pin_load 0.0025 [get_ports {prob_out[5][7]}]
set_load -pin_load 0.0025 [get_ports {prob_out[5][6]}]
set_load -pin_load 0.0025 [get_ports {prob_out[5][5]}]
set_load -pin_load 0.0025 [get_ports {prob_out[5][4]}]
set_load -pin_load 0.0025 [get_ports {prob_out[5][3]}]
set_load -pin_load 0.0025 [get_ports {prob_out[5][2]}]
set_load -pin_load 0.0025 [get_ports {prob_out[5][1]}]
set_load -pin_load 0.0025 [get_ports {prob_out[5][0]}]
set_load -pin_load 0.0025 [get_ports {prob_out[4][7]}]
set_load -pin_load 0.0025 [get_ports {prob_out[4][6]}]
set_load -pin_load 0.0025 [get_ports {prob_out[4][5]}]
set_load -pin_load 0.0025 [get_ports {prob_out[4][4]}]
set_load -pin_load 0.0025 [get_ports {prob_out[4][3]}]
set_load -pin_load 0.0025 [get_ports {prob_out[4][2]}]
set_load -pin_load 0.0025 [get_ports {prob_out[4][1]}]
set_load -pin_load 0.0025 [get_ports {prob_out[4][0]}]
set_load -pin_load 0.0025 [get_ports {prob_out[3][7]}]
set_load -pin_load 0.0025 [get_ports {prob_out[3][6]}]
set_load -pin_load 0.0025 [get_ports {prob_out[3][5]}]
set_load -pin_load 0.0025 [get_ports {prob_out[3][4]}]
set_load -pin_load 0.0025 [get_ports {prob_out[3][3]}]
set_load -pin_load 0.0025 [get_ports {prob_out[3][2]}]
set_load -pin_load 0.0025 [get_ports {prob_out[3][1]}]
set_load -pin_load 0.0025 [get_ports {prob_out[3][0]}]
set_load -pin_load 0.0025 [get_ports {prob_out[2][7]}]
set_load -pin_load 0.0025 [get_ports {prob_out[2][6]}]
set_load -pin_load 0.0025 [get_ports {prob_out[2][5]}]
set_load -pin_load 0.0025 [get_ports {prob_out[2][4]}]
set_load -pin_load 0.0025 [get_ports {prob_out[2][3]}]
set_load -pin_load 0.0025 [get_ports {prob_out[2][2]}]
set_load -pin_load 0.0025 [get_ports {prob_out[2][1]}]
set_load -pin_load 0.0025 [get_ports {prob_out[2][0]}]
set_load -pin_load 0.0025 [get_ports {prob_out[1][7]}]
set_load -pin_load 0.0025 [get_ports {prob_out[1][6]}]
set_load -pin_load 0.0025 [get_ports {prob_out[1][5]}]
set_load -pin_load 0.0025 [get_ports {prob_out[1][4]}]
set_load -pin_load 0.0025 [get_ports {prob_out[1][3]}]
set_load -pin_load 0.0025 [get_ports {prob_out[1][2]}]
set_load -pin_load 0.0025 [get_ports {prob_out[1][1]}]
set_load -pin_load 0.0025 [get_ports {prob_out[1][0]}]
set_load -pin_load 0.0025 [get_ports {prob_out[0][7]}]
set_load -pin_load 0.0025 [get_ports {prob_out[0][6]}]
set_load -pin_load 0.0025 [get_ports {prob_out[0][5]}]
set_load -pin_load 0.0025 [get_ports {prob_out[0][4]}]
set_load -pin_load 0.0025 [get_ports {prob_out[0][3]}]
set_load -pin_load 0.0025 [get_ports {prob_out[0][2]}]
set_load -pin_load 0.0025 [get_ports {prob_out[0][1]}]
set_load -pin_load 0.0025 [get_ports {prob_out[0][0]}]
set_max_delay 11 -from [list \
  [get_ports clk]  \
  [get_ports rst_n]  \
  [get_ports valid_in]  \
  [get_ports {in_data[7][7]}]  \
  [get_ports {in_data[7][6]}]  \
  [get_ports {in_data[7][5]}]  \
  [get_ports {in_data[7][4]}]  \
  [get_ports {in_data[7][3]}]  \
  [get_ports {in_data[7][2]}]  \
  [get_ports {in_data[7][1]}]  \
  [get_ports {in_data[7][0]}]  \
  [get_ports {in_data[6][7]}]  \
  [get_ports {in_data[6][6]}]  \
  [get_ports {in_data[6][5]}]  \
  [get_ports {in_data[6][4]}]  \
  [get_ports {in_data[6][3]}]  \
  [get_ports {in_data[6][2]}]  \
  [get_ports {in_data[6][1]}]  \
  [get_ports {in_data[6][0]}]  \
  [get_ports {in_data[5][7]}]  \
  [get_ports {in_data[5][6]}]  \
  [get_ports {in_data[5][5]}]  \
  [get_ports {in_data[5][4]}]  \
  [get_ports {in_data[5][3]}]  \
  [get_ports {in_data[5][2]}]  \
  [get_ports {in_data[5][1]}]  \
  [get_ports {in_data[5][0]}]  \
  [get_ports {in_data[4][7]}]  \
  [get_ports {in_data[4][6]}]  \
  [get_ports {in_data[4][5]}]  \
  [get_ports {in_data[4][4]}]  \
  [get_ports {in_data[4][3]}]  \
  [get_ports {in_data[4][2]}]  \
  [get_ports {in_data[4][1]}]  \
  [get_ports {in_data[4][0]}]  \
  [get_ports {in_data[3][7]}]  \
  [get_ports {in_data[3][6]}]  \
  [get_ports {in_data[3][5]}]  \
  [get_ports {in_data[3][4]}]  \
  [get_ports {in_data[3][3]}]  \
  [get_ports {in_data[3][2]}]  \
  [get_ports {in_data[3][1]}]  \
  [get_ports {in_data[3][0]}]  \
  [get_ports {in_data[2][7]}]  \
  [get_ports {in_data[2][6]}]  \
  [get_ports {in_data[2][5]}]  \
  [get_ports {in_data[2][4]}]  \
  [get_ports {in_data[2][3]}]  \
  [get_ports {in_data[2][2]}]  \
  [get_ports {in_data[2][1]}]  \
  [get_ports {in_data[2][0]}]  \
  [get_ports {in_data[1][7]}]  \
  [get_ports {in_data[1][6]}]  \
  [get_ports {in_data[1][5]}]  \
  [get_ports {in_data[1][4]}]  \
  [get_ports {in_data[1][3]}]  \
  [get_ports {in_data[1][2]}]  \
  [get_ports {in_data[1][1]}]  \
  [get_ports {in_data[1][0]}]  \
  [get_ports {in_data[0][7]}]  \
  [get_ports {in_data[0][6]}]  \
  [get_ports {in_data[0][5]}]  \
  [get_ports {in_data[0][4]}]  \
  [get_ports {in_data[0][3]}]  \
  [get_ports {in_data[0][2]}]  \
  [get_ports {in_data[0][1]}]  \
  [get_ports {in_data[0][0]}] ] -to [list \
  [get_ports valid_out]  \
  [get_ports {prob_out[7][7]}]  \
  [get_ports {prob_out[7][6]}]  \
  [get_ports {prob_out[7][5]}]  \
  [get_ports {prob_out[7][4]}]  \
  [get_ports {prob_out[7][3]}]  \
  [get_ports {prob_out[7][2]}]  \
  [get_ports {prob_out[7][1]}]  \
  [get_ports {prob_out[7][0]}]  \
  [get_ports {prob_out[6][7]}]  \
  [get_ports {prob_out[6][6]}]  \
  [get_ports {prob_out[6][5]}]  \
  [get_ports {prob_out[6][4]}]  \
  [get_ports {prob_out[6][3]}]  \
  [get_ports {prob_out[6][2]}]  \
  [get_ports {prob_out[6][1]}]  \
  [get_ports {prob_out[6][0]}]  \
  [get_ports {prob_out[5][7]}]  \
  [get_ports {prob_out[5][6]}]  \
  [get_ports {prob_out[5][5]}]  \
  [get_ports {prob_out[5][4]}]  \
  [get_ports {prob_out[5][3]}]  \
  [get_ports {prob_out[5][2]}]  \
  [get_ports {prob_out[5][1]}]  \
  [get_ports {prob_out[5][0]}]  \
  [get_ports {prob_out[4][7]}]  \
  [get_ports {prob_out[4][6]}]  \
  [get_ports {prob_out[4][5]}]  \
  [get_ports {prob_out[4][4]}]  \
  [get_ports {prob_out[4][3]}]  \
  [get_ports {prob_out[4][2]}]  \
  [get_ports {prob_out[4][1]}]  \
  [get_ports {prob_out[4][0]}]  \
  [get_ports {prob_out[3][7]}]  \
  [get_ports {prob_out[3][6]}]  \
  [get_ports {prob_out[3][5]}]  \
  [get_ports {prob_out[3][4]}]  \
  [get_ports {prob_out[3][3]}]  \
  [get_ports {prob_out[3][2]}]  \
  [get_ports {prob_out[3][1]}]  \
  [get_ports {prob_out[3][0]}]  \
  [get_ports {prob_out[2][7]}]  \
  [get_ports {prob_out[2][6]}]  \
  [get_ports {prob_out[2][5]}]  \
  [get_ports {prob_out[2][4]}]  \
  [get_ports {prob_out[2][3]}]  \
  [get_ports {prob_out[2][2]}]  \
  [get_ports {prob_out[2][1]}]  \
  [get_ports {prob_out[2][0]}]  \
  [get_ports {prob_out[1][7]}]  \
  [get_ports {prob_out[1][6]}]  \
  [get_ports {prob_out[1][5]}]  \
  [get_ports {prob_out[1][4]}]  \
  [get_ports {prob_out[1][3]}]  \
  [get_ports {prob_out[1][2]}]  \
  [get_ports {prob_out[1][1]}]  \
  [get_ports {prob_out[1][0]}]  \
  [get_ports {prob_out[0][7]}]  \
  [get_ports {prob_out[0][6]}]  \
  [get_ports {prob_out[0][5]}]  \
  [get_ports {prob_out[0][4]}]  \
  [get_ports {prob_out[0][3]}]  \
  [get_ports {prob_out[0][2]}]  \
  [get_ports {prob_out[0][1]}]  \
  [get_ports {prob_out[0][0]}] ]
set_clock_gating_check -setup 0.0 
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports rst_n]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports valid_in]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[7][7]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[7][6]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[7][5]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[7][4]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[7][3]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[7][2]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[7][1]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[7][0]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[6][7]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[6][6]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[6][5]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[6][4]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[6][3]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[6][2]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[6][1]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[6][0]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[5][7]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[5][6]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[5][5]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[5][4]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[5][3]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[5][2]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[5][1]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[5][0]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[4][7]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[4][6]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[4][5]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[4][4]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[4][3]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[4][2]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[4][1]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[4][0]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[3][7]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[3][6]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[3][5]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[3][4]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[3][3]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[3][2]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[3][1]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[3][0]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[2][7]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[2][6]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[2][5]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[2][4]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[2][3]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[2][2]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[2][1]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[2][0]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[1][7]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[1][6]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[1][5]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[1][4]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[1][3]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[1][2]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[1][1]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[1][0]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[0][7]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[0][6]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[0][5]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[0][4]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[0][3]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[0][2]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[0][1]}]
set_input_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {in_data[0][0]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports valid_out]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[7][7]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[7][6]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[7][5]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[7][4]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[7][3]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[7][2]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[7][1]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[7][0]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[6][7]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[6][6]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[6][5]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[6][4]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[6][3]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[6][2]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[6][1]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[6][0]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[5][7]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[5][6]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[5][5]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[5][4]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[5][3]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[5][2]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[5][1]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[5][0]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[4][7]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[4][6]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[4][5]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[4][4]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[4][3]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[4][2]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[4][1]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[4][0]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[3][7]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[3][6]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[3][5]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[3][4]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[3][3]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[3][2]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[3][1]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[3][0]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[2][7]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[2][6]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[2][5]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[2][4]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[2][3]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[2][2]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[2][1]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[2][0]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[1][7]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[1][6]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[1][5]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[1][4]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[1][3]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[1][2]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[1][1]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[1][0]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[0][7]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[0][6]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[0][5]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[0][4]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[0][3]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[0][2]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[0][1]}]
set_output_delay -clock [get_clocks clk] -add_delay 3.0 [get_ports {prob_out[0][0]}]
set_max_fanout 16.000 [current_design]
set_max_transition 0.35 [current_design]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports clk]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports rst_n]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports valid_in]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[7][7]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[7][6]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[7][5]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[7][4]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[7][3]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[7][2]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[7][1]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[7][0]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[6][7]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[6][6]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[6][5]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[6][4]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[6][3]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[6][2]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[6][1]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[6][0]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[5][7]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[5][6]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[5][5]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[5][4]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[5][3]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[5][2]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[5][1]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[5][0]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[4][7]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[4][6]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[4][5]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[4][4]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[4][3]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[4][2]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[4][1]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[4][0]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[3][7]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[3][6]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[3][5]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[3][4]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[3][3]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[3][2]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[3][1]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[3][0]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[2][7]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[2][6]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[2][5]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[2][4]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[2][3]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[2][2]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[2][1]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[2][0]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[1][7]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[1][6]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[1][5]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[1][4]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[1][3]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[1][2]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[1][1]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[1][0]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[0][7]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[0][6]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[0][5]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[0][4]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[0][3]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[0][2]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[0][1]}]
set_driving_cell -lib_cell BUF_X6M_A9TR -library sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c -pin "Y" [get_ports {in_data[0][0]}]
set_ideal_network [get_ports clk]
set_ideal_network [get_ports rst_n]
set_wire_load_mode "top"
set_clock_uncertainty -setup 0.125 [get_clocks clk]
set_clock_uncertainty -hold 0.125 [get_clocks clk]
