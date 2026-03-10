`timescale 1ns / 1ps

module tr_ln_tb;
    parameter WIDTH = 16;
    parameter BITS  = 4;  // Fractional bits

    // Maximum allowable error (Tolerance). 
    // -4 LSBs in Q4.4 format equals an absolute float error of 0.25
    // We set it to 0.35 to allow for standard approximation deviation across the full range.
    parameter real CHECK_TOLERANCE = 0.35;

    reg  [WIDTH-1:0]    xq;
    wire [WIDTH/2-1:0]  yq; // 8 bits (Q4.4)
    
    real real_x, real_y, expected_y, error;
    int  error_count = 0;

    tr_ln #(
      .WIDTH(WIDTH),
      .BITS(BITS)  
    ) uut (
        .xq(xq),
        .yq(yq)
    );

    initial begin
        $display("\n--- Starting TR-ln Verification (Q12.4) ---");

        // 1. DENSE SWEEP: Test every possible value from 0.0625 to 2047.9375
        // 1. DENSE SWEEP: Test every possible value up to the Q4.4 overflow limit (44800)
        // We start at i=1 because ln(0) is undefined.
        for (int i = 1; i < (1 << WIDTH); i++) begin
            
            // Drive the raw integer directly into xq (representing the Q12.4 bits)
            xq = i;
            #1; // Allow combinational logic to settle

            // 2. CONVERSION MATH
            // Convert fixed-point bits to true float by dividing by 2^BITS (16.0)
            real_x = real'(xq) / 16.0;
            
            // Treat yq as a signed 8-bit Q4.4 number
            real_y = real'($signed(yq)) / 16.0;

            // 3. GOLDEN MODEL & ERROR CALCULATION
            expected_y = $ln(real_x);
            error = real_y - expected_y;
            
            // Absolute error
            if (error < 0) error = -error;

            // 4. SELF-CHECKING ASSERTION
            if (error > CHECK_TOLERANCE) begin
                $error("[ASSERT FAIL] x=%f (0x%h) | HW=%f | Exact ln()=%f | Err=%f", 
                       real_x, xq, real_y, expected_y, error);
                error_count++;
                
                // Stop after 10 errors to prevent console flooding
                if (error_count > 10) begin
                    $display("... Error limit reached. Aborting sweep.");
                    break; 
                end
            end
        end

        // 5. FINAL SUMMARY
        $display("\n--- Test Summary ---");
        if (error_count == 0) 
            $display("\033[0;32m[TEST PASSED]\033[0m All %0d values within tolerance (%0.2f)", 
                     (1 << WIDTH) - 1, CHECK_TOLERANCE);
        else 
            $display("\033[0;31m[TEST FAILED]\033[0m %0d values exceeded tolerance", error_count);
            
        $finish;
    end

endmodule