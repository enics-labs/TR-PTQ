# 1. Ensure definitions are loaded
source ../inputs/vec_mac_su.defines -quiet

# 2. Re-read HDL and Elaborate (Resetting the design to RTL state)
# We do this to clear the previous "synthesized" state from memory
read_hdl -language sv -f $design(read_hdl_list)
elaborate $design(TOPLEVEL)

# 3. Create the Snapshot (CRITICAL STEP)
# This creates the file your loop will try to read 10+ times
set snapshot_path "[pwd]/elaborated_snapshot.db"
write_db $design(TOPLEVEL) -to_file $snapshot_path

puts "Snapshot created. Now you can paste the loop code."

##############################################################
# PHASE 2: THE OPTIMIZATION LOOP (Quiet Version)
##############################################################

# --- QUIET MODE SETTINGS ---
set_db source_verbose false
set_db information_level 0 

set start_period  10
set end_period    2
set step          1
set last_passing_period $start_period

for {set clk_p $start_period} {$clk_p >= $end_period} {set clk_p [expr $clk_p - $step]} {
    
    puts "\n=============================================="
    puts "  Starting Iteration with Clock Period: $clk_p ns"
    puts "==============================================\n"

    # A. CLEANUP 
    if {[get_db designs] != ""} {
        redirect /dev/null { delete_obj [get_db designs] }
    }

    # B. RESTORE CLEAN STATE
    redirect /dev/null { read_db $snapshot_path }

    # C. SET CONTEXT
    current_design $design(TOPLEVEL)

    # D. UPDATE VARIABLES 
    set design(CLK_PERIOD) $clk_p
    set design(clock_period_list) [list $clk_p]

    set design(INPUT_DELAY)       [expr $clk_p / 4.0]
    set design(OUTPUT_DELAY)      [expr $clk_p / 4.0]
    set design(INPUT_TRANSITION)  [expr $clk_p / 10.0]

    # E. SOURCE THE SDC
    source $design(functional_sdc)

    # F. RUN SYNTHESIS (The Noisy Part)
    ##########################
    #     Synthesize
    ##########################
    enics_start_stage "synthesis"

    # Set Synthesis Efforts
    set_db syn_generic_effort low
    set_db syn_map_effort low
    set_db syn_opt_effort low
    suppress_messages "ST-110 ST-112"

    redirect /dev/null {
        # Don't use scan cells
        enics_message "Setting Don't Use on scan flip flops"
        foreach cell [get_db lib_cells -if {.scan_enable_pins!=""}] {set_db $cell .avoid true}

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
                syn_opt -physical
            } else {
                syn_opt 
            }
        }
    }

    # G. CHECK TIMING
    set worst_paths [report_timing -max_paths 1 -nworst 1 -collection]
    
    if {$worst_paths != ""} {
        set worst_path [lindex $worst_paths 0]
        set slack [get_db $worst_path .slack]
    } else {
        set slack 0.0
    }

    # 2. Check DRVs (Max Transition / Cap / Fanout)
    set drv_count 0
    set rpt_file "temp_drv_check.rpt"
    
    # 1. Run the report and capture it to a file
    #    We check Transition, Capacitance, and Fanout
    redirect $rpt_file {
        puts "Checking Max Transition..."
        report_constraint -drv_violation_type max_transition -all_violators
        puts "Checking Max Capacitance..."
        report_constraint -drv_violation_type max_capacitance -all_violators
        puts "Checking Max Fanout..."
        report_constraint -drv_violation_type max_fanout -all_violators
    }

    # 2. Open the file and parse it using Regex
    if {[file exists $rpt_file]} {
        set fp [open $rpt_file r]
        set file_data [read $fp]
        close $fp
        
        # Regex explains: Look for "violation total = <number>"
        # This matches the text shown in your screenshot.
        set matches [regexp -all -inline {violation total\s*=\s*(\d+)} $file_data]
        
        # matches list format: { "violation total = 25" "25" ... }
        foreach {whole_match count} $matches {
            set drv_count [expr $drv_count + $count]
        }
        
        # (Optional) Clean up temp file
        file delete $rpt_file
    }

    puts "  -> Result at ${clk_p}ns: Slack = $slack | DRV Violations = $drv_count"

    # H. DECISION LOGIC
    if {$slack < 0.0} {
        puts "\n*************************************************"
        puts "  FAILURE: Negative Slack ($slack) at ${clk_p}ns."
        puts "  Stopping loop."
        puts "*************************************************"
        break

    } elseif {$drv_count > 0} {
        puts "\n*************************************************"
        puts "  FAILURE: Found $drv_count DRV violations at ${clk_p}ns."
        puts "  (Slack was $slack, but design rules failed)."
        puts "  Stopping loop."
        puts "*************************************************"
        break

    } else {
        # SUCCESS: Positive Slack AND Zero DRVs
        puts "  >> SUCCESS: Clean run. Updating best period."
        set last_passing_period $clk_p
        set found_clean_run true

        write_db -all_root_attributes -to_file "optimal_run.db"
    }
}

##############################################################
# PHASE 3: FINAL EXECUTION (The Addition)
##############################################################

puts "\n=============================================="
puts "  Running FINAL Synthesis at: $last_passing_period ns"
puts "==============================================\n"

# 1. Clean up the failed run
if {[get_db designs] != ""} {
    redirect /dev/null { delete_obj [get_db designs] }
}

# 2. Restore clean DB
redirect /dev/null { read_db $snapshot_path }
current_design $design(TOPLEVEL)

# 3. Setup variables for the BEST period (Not the current failing loop variable)
set design(CLK_PERIOD) $last_passing_period
set design(clock_period_list) [list $last_passing_period]

set design(INPUT_DELAY)       [expr $last_passing_period / 4.0]
set design(OUTPUT_DELAY)      [expr $last_passing_period / 4.0]
set design(INPUT_TRANSITION)  [expr $last_passing_period / 10.0]

# 4. Source SDC
source $design(functional_sdc)

# 5. Run Final Synthesis
##########################
#     Synthesize
##########################
enics_start_stage "synthesis"

# Set Synthesis Efforts
set_db syn_generic_effort low
set_db syn_map_effort low
set_db syn_opt_effort low
suppress_messages "ST-110 ST-112"


redirect /dev/null {
    # Don't use scan cells
    enics_message "Setting Don't Use on scan flip flops"
    foreach cell [get_db lib_cells -if {.scan_enable_pins!=""}] {set_db $cell .avoid true}

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
            syn_opt -physical
        } else {
            syn_opt 
        }
    }
}

puts "  -> Final Synthesis Complete. Ready for export."

# Restore verbosity at the end
set_db source_verbose true
set_db information_level 1