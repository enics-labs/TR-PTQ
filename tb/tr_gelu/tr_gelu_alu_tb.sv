`timescale 1ns/1ps

module tr_gelu_tb;

    localparam int W = 8;
    localparam int FRAC_W = 4;
    localparam int NUM_TESTS = 59;
    localparam real CHECK_TOLERANCE = 0.15;

    // Signals
    logic clk, rst_n, valid_in, valid_out, mode;
    logic signed [W-1:0] x_in, gelu_out;
    logic [7:0] inv_s_in;

    // Test Vectors
    logic signed [W-1:0] test_vectors [NUM_TESTS];
    logic [7:0]          pass1_results [NUM_TESTS]; // Holds inv_S

    int total_errors = 0;

    // DUT
    tr_gelu #(.W(W)) dut (.*);

    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Simulation Flow
    initial begin
        rst_n = 0; valid_in = 0; mode = 0; x_in = 0; inv_s_in = 0;
        
        // 1. Generate Test Vectors
        test_vectors[0] = 8'd0;   test_vectors[1] = 8'd16;  test_vectors[2] = -8'd16;
        test_vectors[3] = 8'd32;  test_vectors[4] = -8'd32; test_vectors[5] = 8'd64;
        test_vectors[6] = -8'd64; test_vectors[7] = 8'd127; test_vectors[8] = -8'd128;
        
        for (int i=9; i<NUM_TESTS; i++) begin
            test_vectors[i] = $random;
        end

        repeat(3) @(posedge clk);
        rst_n = 1;
        $display("\n=========================================================================");
        $display("                 STARTING TWO-PASS TR-GELU SIMULATION                    ");
        $display("=========================================================================");

        // ========================================================================
        // EXECUTE PASS 1: MODE 0 (Calculate inv_S)
        // ========================================================================
        fork
            begin // Stimulus Driver
                for (int i=0; i<NUM_TESTS; i++) begin
                    x_in <= test_vectors[i]; mode <= 0; valid_in <= 1; @(posedge clk);
                end
                valid_in <= 0;
            end
            begin // Result Catcher
                for (int i=0; i<NUM_TESTS; i++) begin
                    do begin @(posedge clk); end while (!valid_out);
                    pass1_results[i] = gelu_out; 
                end
            end
        join

        $display("--> PASS 1 (Reciprocals) Generated Successfully.");
        repeat(10) @(posedge clk);

        // ========================================================================
        // EXECUTE PASS 2: MODE 1 (Calculate Final GELU)
        // ========================================================================
        fork
            begin // Stimulus Driver
                for (int i=0; i<NUM_TESTS; i++) begin
                    x_in     <= test_vectors[i];
                    inv_s_in <= pass1_results[i]; // Feed back the SRAM data
                    mode     <= 1; 
                    valid_in <= 1; 
                    @(posedge clk);
                end
                valid_in <= 0;
            end
            begin // Result Catcher & Checker
                real x_real, exp_val, hw_val, err;
                
                for (int i=0; i<NUM_TESTS; i++) begin
                    do begin @(posedge clk); end while (!valid_out);
                    
                    x_real  = real'(test_vectors[i]) / 16.0;
                    exp_val = x_real / (1.0 + $exp(-1.702 * x_real));
                    hw_val  = real'(gelu_out) / 16.0;
                    
                    err = (hw_val > exp_val) ? (hw_val - exp_val) : (exp_val - hw_val);

                    if (err <= CHECK_TOLERANCE)
                        $display("[TEST %2d] \033[0;32mPASSED\033[0m | x_in: %4d | Exp: %6.3f | HW: %6.3f", 
                                 i, test_vectors[i], exp_val, hw_val);
                    else begin
                        $display("[TEST %2d] \033[0;31mFAILED\033[0m | x_in: %4d | Exp: %6.3f | HW: %6.3f | Err: %6.3f", 
                                 i, test_vectors[i], exp_val, hw_val, err);
                        total_errors++;
                    end
                end
            end
        join

        $display("=========================================================================");
        if (total_errors == 0)
            $display("\033[0;32m[TEST PASSED]\033[0m End-to-End Two-Pass Execution Verified!");
        else
            $display("\033[0;31m[TEST FAILED]\033[0m %0d errors found.", total_errors);
            
        $finish;
    end
endmodule