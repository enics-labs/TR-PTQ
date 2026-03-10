`timescale 1ns/1ps

module tr_div_tb;

    localparam int M = 16;
    localparam int K = 4;
    localparam int ITER = 2; // Match the module default

    // Tolerance allowance for combining TR-ln and TR-exp errors
    localparam real CHECK_TOLERANCE = 0.08; 

    logic        clk = 0;
    logic        rst_n = 0;

    logic [15:0] xq;
    logic [7:0]  yq_rtl;

    real real_x, real_y, expected_y, error;
    int error_count = 0;

    // DUT
    tr_reciprocal #(
        .WIDTH(16),
        .ITER (ITER),
        .BITS (K)
    ) dut (
        .clk  (clk),
        .rst_n(rst_n),
        .xq   (xq),
        .yq   (yq_rtl)
    );

    always #5 clk = ~clk;

    initial begin
        $display("\n--- Starting TR-Reciprocal Verification (1/x) ---");
        rst_n = 0;
        #20 rst_n = 1;

        // Sweep positive inputs starting from 1.0 (16 in Q12.4)
        // We test up to 100.0 (1600 in Q12.4)
        for (int i = 16; i <= 1600; i++) begin
            
            xq = i;
            #1; // Allow purely combinational logic to settle

            // Conversion Math
            real_x = real'(xq) / 16.0;
            
            // The output of tr_exp has an inherent *256 scaling on e_a, and *16 on mantisa.
            // After shifting right by 4, the output is scaled by 2^8 (256) (Q0.8 format)
            real_y = real'(yq_rtl) / 256.0;

            // Golden model: 1/x
            expected_y = 1.0 / real_x;
            error = real_y - expected_y;
            
            if (error < 0) error = -error; // Absolute error

            if (error > CHECK_TOLERANCE) begin
                $error("[ASSERT FAIL] x=%f | HW_1/x=%f | Exact_1/x=%f | Err=%f", 
                          real_x, real_y, expected_y, error);
                error_count++;
                
                if (error_count > 10) begin
                    $display("... Error limit reached. Aborting sweep.");
                    break;
                end
            end
        end

        $display("\n--- Test Summary ---");
        if (error_count == 0) 
            $display("\033[0;32m[TEST PASSED]\033[0m Hardware successfully approximates 1/x within tolerance.");
        else 
            $display("\033[0;31m[TEST FAILED]\033[0m %0d values exceeded tolerance", error_count);

        $finish;
    end

endmodule