
set_library_unit -time 1ns -cap 1pf

# Set on-chip variation analysis
set_db timing_analysis_type ocv
set_db timing_analysis_cppr both

# From RAK
set_db timing_analysis_cppr both



# From Negev - check
set_db write_db_auto_save_user_globals true
set_db timing_report_unconstrained_paths true
set_db timing_cppr_threshold_ps 5
# Ports without input delay are unconstrained by default
set_db timing_apply_default_primary_input_assertion false
set_db timing_enable_si_cppr                        true
# set_db timing_remove_clock_reconvergence_pessimism true
set_db timing_disable_inferred_clock_gating_checks  true
set_db delaycal_equivalent_waveform_model propagation
# Default false; switch on in routing
set_db delaycal_enable_si false
# To see incr delay separetly in reports
set_db si_delay_separate_on_data true
# For correlation to Tempus
set_db si_glitch_enable_report true
#Allow swapping between chaines of the same partition
set_db reorder_scan_allow_swapping true
# set_db timing_analysis_aocv               true
# set_db timing_enable_aocv_slack_based     true
# set_db timing_aocv_analysis_mode          launch_capture; #{launch_capture | clock_only | separate_data_clock | combine_launch_capture}
# set_db timing_extract_model_aocv_mode     graph_based
# set_db timing_aocv_derate_mode            aocv_additive;  #{aocv_multiplicative | aocv_additive}

## Global Placement Settings
###########################################################
#set_db place_global_place_io_pins false
#set_db place_detail_no_filler_without_implant true
#set_db place_detail_use_no_diffusion_one_site_filler true

# CCopt
# set_db cts_target_max_transition_time_top   $clk_slew
# set_db cts_target_max_transition_time_trunk $clk_slew
# set_db cts_target_max_transition_time_leaf  $clk_slew



#### Routing
#set_db route_design_detail_auto_stop                     false
#set_db route_design_concurrent_minimize_via_count_effort high
#set_db route_design_strict_honor_route_rule              wire
#set_db route_design_with_via_in_pin "1:2"
#set_db route_design_stripe_layer_range "$nonDPT_MinRouteLayer:$MaxRouteLayer"
#set_db route_design_tie_net_to_shape "stripe"
#if { [llength $antenna_cell]>0 } {
#    set_db route_design_antenna_diode_insertion true
#    set_db route_design_antenna_cell_name $antenna_cell
#}
### don't use pin as a jumper - make one contact
#set_db route_design_allow_pin_as_feedthru false
## don't taper to the output pin causing EM issues
#set_db route_design_detail_no_taper_on_output_pin true
#




