`timescale 1ns/1ps

module tr_exp_tb;

    parameter int FRAC = 4;
    parameter int ITER = 2;
    parameter real CLK_PERIOD = 10;
    // Maximum allowable difference between HW and Math (approx 10% for Taylor)
    parameter real CHECK_TOLERANCE = 0.070; 

    logic clk, rst_n;
    logic signed [7:0] x;

    // --------------------------------- //
    // CHANGE 1
    // The tr_exp module outputs: e_a, mantisa, is_zero.
    // Original TB: 
    //      mantisa and is_zero are missing so the (.*) in the dut instantiation may not be fully connected.
    //      Additionally, the y was not connected since there is no y output, so all values "from hardware" were 0
    // Updated TB:
    //      dut is connected properly, and the exponent y is calculated.
    //
    // CHANGE 2
    // The for loop sweep range is to big, and the stop condition may be inaccurate
    // Original TB:
    //      The sweep ranges from 0 to -512, when reaching -256 (-256/32 = -8 = 1000.0000) at -257 there is overflow (0111.1111 = +7.9375)
    //      So we get a positive value which breaks our bounding condition of x <= 0.
    //      The break condition was set to 'if (real_x < -8), since -8.0 is not less than -8 we keep going and overflow to positive x.
    // Updated TB:
    //      The sweep range is updated: from 0 to -256, and the break condition is updated to 'if (real_x <= -8.0)'.
    //
    // CHANGE 3
    // Added average error(%) calculation
    // NOTE: Just the Avg is not so helpful, maybe add min, max, and stdev?
    //
    // ACTUAL POTENTIAL PROBLEM:
    // Index 7 of the LUT is hardcoded to hold e^0 (255). When x is below -7.5625 (1000.0110) the rounding bit is zero.
    // And the integer is -8 (1000) so the fliped_rounded_int (from round.sv) is 111 which takes index 7 of the LUT.
    // This results in x ranges -7.5625 to -8.0 to get exponent value of e^0.
    // --------------------------------- //

    // DUT Outputs    
    logic [7:0]  e_a;      // Q4
    logic [7:0] mantisa;
    logic       is_zero;

    logic [7:0] y;

    // Calculate the final product and shift back by FRAC
    wire [15:0] mult_result = e_a * mantisa;
    assign y = mult_result >> FRAC;

    // Internal tracking
    real real_x, real_y, expected_y, error;
    int error_count = 0;
    real error_accum  = 0;

    int i;

    import "DPI-C" function int dpi_real_to_qmk(
        input real real_val,
        input int  M,
        input int  K
    );
    import "DPI-C" function real dpi_qmk_to_real(
        input int fixed_val,
        input int K
    );

    tr_exp #(.FRAC(FRAC), .ITER(ITER)) dut (
            .x(x),
            .e_a(e_a),
            .mantisa(mantisa),
            .is_zero(is_zero)
        );

    
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
        for (i = 0; i >= -256; i--) begin
            @(negedge clk);
            x = dpi_real_to_qmk(i/32.0, 4, 4);
            @(posedge clk); 
            #1; // Allow logic to settle

            // Conversion Math
            real_x     = dpi_qmk_to_real(x, 4);
            real_y     = dpi_qmk_to_real(y, 8);

            // Stop condition (redundant)
            if (real_x <= -8.0)
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
            error_accum += error;
        
        end

        $display("\n--- Test Summary ---");
        if (error_count == 0) begin
            $display("SUCCESS: All values within tolerance (%0.2f)", CHECK_TOLERANCE);
            $display("Average error: %0.2f%", (error_accum / $abs(i)) * 100);
        end else 
            $display("FAILURE: %0d values exceeded tolerance", error_count);
            
        $finish;
    end

endmodule