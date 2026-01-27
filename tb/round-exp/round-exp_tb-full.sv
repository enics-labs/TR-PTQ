`timescale 1ns/1ps

module tr_exp_tb;

    parameter int FRAC = 4;
    parameter int ITER = 2;
    parameter real CLK_PERIOD = 10;
    // Maximum allowable difference between HW and Math (approx 10% for Taylor)
    parameter real CHECK_TOLERANCE = 0.070; 

    logic clk, rst_n;
    logic signed [7:0] x;
    logic [7:0] y;
    logic [7:0]  e_a;      // Q4
    logic [3:0]  x_frac;

    // Internal tracking
    real real_x, real_y, expected_y, error;
    int error_count = 0;

    import "DPI-C" function int dpi_real_to_qmk(
        input real real_val,
        input int  M,
        input int  K
    );
    import "DPI-C" function real dpi_qmk_to_real(
        input int fixed_val,
        input int K
    );

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
        for (int i = 0; i >= -512; i--) begin
            @(negedge clk);
            x = dpi_real_to_qmk(i/32.0, 4, 4);
            @(posedge clk); 
            #1; // Allow logic to settle

            // Conversion Math
            real_x     = dpi_qmk_to_real(x, 4);
            real_y     = dpi_qmk_to_real(y, 8);
            if (real_x <-8)
                break;
            // dpi_real_to_qmk(i/32.0, 4, 4);
            expected_y = $exp(real_x);
            error      = (real_y - expected_y);
            // Absolute error
            if (error < 0) error = -error;
            
            // $display("$%0t| [DEBUG] x:%b(%0f)| y:%h(%0f): $exp(real_x): %0f| diff:%0f| e_a:%0d| x_frac: %0x(%0b)",$time, x, real_x, y, real_y, $exp(real_x), real_y - $exp(real_x), e_a, x_frac, x_frac);

            // Self-Checking Logic
            if (error > CHECK_TOLERANCE) begin
                $error("[ASSERT FAIL] x=%f | HW=%f | Exp=%f | Err=%f", 
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