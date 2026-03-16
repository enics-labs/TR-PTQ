# ==============================================================================
# FMAX SWEEP PROCEDURE FOR CADENCE GENUS
# ==============================================================================

proc run_clock_sweep {start_period end_period step_size clk_port_name reports_dir} {
    
    puts "====================================================================="
    puts " Starting Clock Sweep: $start_period ps down to $end_period ps"
    puts "====================================================================="

    # Open a CSV file to log the Pareto curve results
    set file_path "$reports_dir/clock_sweep_results.csv"
    set csv_file [open $file_path w]
    puts $csv_file "Target_Period_ps, WNS_ps, Total_Area_um2, Leakage_Power_nW"

    puts "\n---> Performing initial technology mapping (syn_map)..."
    # Set the baseline relaxed clock for the initial mapping
    catch {remove_clock core_clk}
    create_clock -name core_clk -period $start_period [get_ports $clk_port_name]
    
    set_input_delay 0.0 -clock core_clk [all_inputs]
    set_output_delay 0.0 -clock core_clk [all_outputs]

    # Map from generic logic to standard cells
    syn_map

    # Loop from the relaxed clock down to the aggressive clock
    for {set period $start_period} {$period >= $end_period} {set period [expr $period - $step_size]} {
        
        puts "\n---> Synthesizing for Target Period: $period ps"

        redirect /dev/null {
            catch {remove_clock core_clk}

            create_clock -name core_clk -period $period [get_ports $clk_port_name]

            set_input_delay 0.0 -clock core_clk [all_inputs]
            set_output_delay 0.0 -clock core_clk [all_outputs]
            
            syn_opt
        }

        # 3. Extract Metrics using Genus Database (get_db)
        # Get Worst Negative Slack (WNS)
        set paths [get_db timing_paths]

        # Check if a path was actually found
        if {[llength $paths] > 0} {
            # Grab the first path in the list and extract its slack
            set worst_path [lindex $paths 0]
            set wns [get_db $worst_path .slack]
        } else {
            set wns 0.0
        }

        # Get Area and Power
        set area  [get_db [current_design] .area]
        set power [get_db [current_design] .leakage_power]

        # 4. Log to CSV and Console
        puts $csv_file "$period, $wns, $area, $power"
        puts "     Result -> WNS: $wns | Area: $area | Leakage: $power"

        # 5. Early Exit Condition (Optional)
        # If WNS goes heavily negative (e.g., failed by more than 15% of the clock), 
        # the architecture simply cannot run this fast. We can break early to save time.
        if {$wns < [expr -0.15 * $period]} {
            puts "\n---> WNS degraded significantly ($wns ps). Stopping sweep early."
            break
        }
    }

    close $csv_file
    puts "\n====================================================================="
    puts " Sweep Complete! Results saved to: $file_path"
    puts "====================================================================="
}