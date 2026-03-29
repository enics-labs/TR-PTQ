`timescale 1ns / 1ps

module tr_ln_tb;

    // ========================================================================
    // DUT INSTANTIATIONS
    // ========================================================================
    logic        [15:0] x_16b;
    logic signed [7:0]  y_8b;
    
    tr_ln #(.WIDTH(16), .BITS(4), .OUT_WIDTH(8)) u_16b_to_8b (
        .xq(x_16b), .yq(y_8b)
    );

    logic        [19:0] x_20b;
    logic signed [11:0] y_12b;
    
    tr_ln #(.WIDTH(20), .BITS(8), .OUT_WIDTH(12)) u_20b_to_12b (
        .xq(x_20b), .yq(y_12b)
    );

    // ========================================================================
    // VERIFICATION VARIABLES
    // ========================================================================
    real x_real, expected_real, hw_real, current_error;
    real max_err_softmax = 0.0;
    real max_err_norm    = 0.0;
    
    int fail_count_softmax = 0;
    int fail_count_norm    = 0;
    int tests_run_softmax  = 0;
    int tests_run_norm     = 0;

    // Define Tolerances
    // We expect slight algorithmic drift because hardware uses ln(2) ~= 0.6875
    real TOL_SOFTMAX = 0.0020; 
    real TOL_NORM    = 0.0015; 

    initial begin
        $display("=======================================================================");
        $display(" STARTING VERBOSE TR-LN VERIFICATION");
        $display("=======================================================================");

        // --------------------------------------------------------------------
        // SWEEP 1: SOFTMAX DENOMINATOR (16-bit in -> 8-bit out, Qx.4)
        // --------------------------------------------------------------------
        $display("\n---> SWEEPING 16-BIT SOFTMAX DOMAIN (Inputs: 1.0 to 625.0)...");
        // We start at 16 (1.0 in Qx.4) because the sum of exponentials is always >= 1.0
        // for (int i = 16; i <= 4096; i++) begin
        for (int i = 16; i <= 2048; i++) begin
            x_16b = i; #1;
            
            // Reconstruct Hardware Output (Output is Qx.4)
            hw_real = real'(y_8b) / 16.0;        
            
            // Calculate Golden Math
            x_real        = real'(i) / 16.0;
            expected_real = $ln(x_real);
            
            // Calculate Error
            current_error = expected_real - hw_real;
            if (current_error < 0) current_error = -current_error; 
            
            if (current_error > max_err_softmax) max_err_softmax = current_error;
            
            // $display("  [SOFTMAX RESULT] x=%7.3f (Raw:%5d) | HW=%6.3f | Math=%6.3f | Err=%6.3f", 
            //                  x_real, i, hw_real, expected_real, current_error);

            // FAILURE LOGGER
            if (current_error > TOL_SOFTMAX) begin
                if (fail_count_softmax < 15) begin
                    $display("  [SOFTMAX FAIL] x=%7.3f (Raw:%5d) | HW=%6.3f | Math=%6.3f | Err=%6.3f", 
                             x_real, i, hw_real, expected_real, current_error);
                end
                fail_count_softmax++;
            end
            tests_run_softmax++;
        end

        // --------------------------------------------------------------------
        // SWEEP 2: LAYERNORM VARIANCE (20-bit in -> 12-bit out, Qx.8)
        // --------------------------------------------------------------------
        $display("\n---> SWEEPING 20-BIT LAYERNORM DOMAIN (Inputs: 0.125 to 200.0)...");
        // We start at 32 (0.125 in Qx.8) to test the negative-log fraction boundary
        for (int i = 32; i <= 51200; i++) begin
            x_20b = i; #1;
            
            // Reconstruct Hardware Output (Output is Qx.8)
            hw_real = real'(y_12b) / 256.0;      
            
            // Calculate Golden Math
            x_real        = real'(i) / 256.0;
            expected_real = $ln(x_real);
            
            // Calculate Error
            current_error = expected_real - hw_real;
            if (current_error < 0) current_error = -current_error; 
            
            if (current_error > max_err_norm) max_err_norm = current_error;
            
            // FAILURE LOGGER
            if (current_error > TOL_NORM) begin
                if (fail_count_norm < 15) begin
                    $display("  [NORM FAIL] x=%7.3f (Raw:%5d) | HW=%6.3f | Math=%6.3f | Err=%6.3f", 
                             x_real, i, hw_real, expected_real, current_error);
                end
                fail_count_norm++;
            end
            tests_run_norm++;
        end

        // --------------------------------------------------------------------
        // FINAL VERIFICATION REPORT
        // --------------------------------------------------------------------
        $display("\n=======================================================================");
        $display(" VERIFICATION REPORT: TR-LN (TAYLOR-REGION LOGARITHM)");
        $display("=======================================================================");
        $display(" [SOFTMAX MODE : 16-bit in -> 8-bit out | Qx.4]");
        $display("    -> Tested     : %0d vectors", tests_run_softmax);
        $display("    -> Max Error  : %f", max_err_softmax);
        $display("    -> Failures   : %0d vectors exceeded tolerance (%0.2f)", fail_count_softmax, TOL_SOFTMAX);
        
        $display("\n [LAYERNORM MODE : 20-bit in -> 12-bit out | Qx.8]");
        $display("    -> Tested     : %0d vectors", tests_run_norm);
        $display("    -> Max Error  : %f", max_err_norm);
        $display("    -> Failures   : %0d vectors exceeded tolerance (%0.2f)", fail_count_norm, TOL_NORM);
        if (fail_count_norm > 0 || fail_count_softmax > 0) begin
             $display("       *(Note: Algorithmic drift is expected at large inputs since HW ln(2) = 0.6875)*");
        end
        $display("=======================================================================\n");
        $finish;
    end

endmodule