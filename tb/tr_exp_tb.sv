`timescale 1ns/1ps

module tr_exp_tb;

    parameter int FRAC = 4;
    parameter int ITER = 2;
    parameter real CLK_PERIOD = 10;
    // Maximum allowable difference between HW and Math (approx 10% for Taylor)
    parameter real CHECK_TOLERANCE = 0.10; 

    logic clk, rst_n;
    logic signed [7:0] x;
    logic [7:0] y;

    // Internal tracking
    real real_x, real_y, expected_y, error;
    int error_count = 0;

    tr_exp #(.FRAC(FRAC), .ITER(ITER)) dut (.*);

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    initial begin
        // Reset
        rst_n = 0;
        x = 0;
        @(posedge clk);
        rst_n = 1;
        
        $display("\n--- Starting Taylor-Region Test (ITER=%0d) ---", ITER);
        
        // Sweep through all negative values and zero
        for (int i = 0; i >= -128; i--) begin
            @(negedge clk);
            x = i[7:0];
            
            @(posedge clk); 
            #1; // Allow logic to settle

            // Conversion Math
            real_x     = real'(x) / 16.0;
            real_y     = real'(y) / 256.0;
            expected_y = $exp(real_x);
            error      = (real_y - expected_y);
            // Absolute error
            if (error < 0) error = -error;

            // Self-Checking Logic
            if (error > CHECK_TOLERANCE) begin
                $display("[ASSERT FAIL] x=%f | HW=%f | Exp=%f | Err=%f", 
                          real_x, real_y, expected_y, error);
                error_count++;
            end
        end

        $display("\n--- Test Summary ---");
        if (error_count == 0) 
            $display("SUCCESS: All values within tolerance (%0.2f)", CHECK_TOLERANCE);
        else 
            $display("FAILURE: %0d values exceeded tolerance", error_count);
            
        $finish;
    end

endmodule