`timescale 1ns/1ps

module parameterized_tr_exp_tb();

    logic signed [7:0] x_in;

    // ========================================================================
    // DUT INSTANTIATIONS (Always 2nd-Order)
    // ========================================================================
    
    // 4-Bit LUT
    logic [3:0] ea_4b; logic [7:0] man_4b; logic z_4b;
    parameterized_tr_exp #(.EA_W(4), .ITER(2)) u_4b (.x(x_in), .e_a(ea_4b), .mantisa(man_4b), .is_zero(z_4b));

    // 8-Bit LUT
    logic [7:0] ea_8b; logic [7:0] man_8b; logic z_8b;
    parameterized_tr_exp #(.EA_W(8), .ITER(2)) u_8b (.x(x_in), .e_a(ea_8b), .mantisa(man_8b), .is_zero(z_8b));

    // 16-Bit LUT
    logic [15:0] ea_16b; logic [7:0] man_16b; logic z_16b;
    parameterized_tr_exp #(.EA_W(16), .ITER(2)) u_16b (.x(x_in), .e_a(ea_16b), .mantisa(man_16b), .is_zero(z_16b));

    // ========================================================================
    // VERIFICATION VARIABLES
    // ========================================================================
    real x_real, expected_real;
    real hw_4, hw_8, hw_16;
    real err_4, err_8, err_16;
    
    real max_err_4 = 0.0, max_err_8 = 0.0, max_err_16 = 0.0;
    real sum_err_4 = 0.0, sum_err_8 = 0.0, sum_err_16 = 0.0;

    int tests_run = 0;

    initial begin
        $display("=======================================================================");
        $display(" LUT PRECISION TRADE-OFF TEST (4-bit vs 8-bit vs 16-bit)");
        $display("=======================================================================");

        // Sweep the entire negative Q4.4 domain (-8.0 to 0)
        for (int i = -128; i <= 0; i++) begin
            x_in = i;
            #1;
            
            x_real = real'(i) / 16.0;
            expected_real = $exp(x_real);
            
            // Reconstruct Hardware Outputs
            hw_4  = (real'(ea_4b)  / 16.0)    * (real'(man_4b)  / 16.0);
            hw_8  = (real'(ea_8b)  / 256.0)   * (real'(man_8b)  / 16.0);
            hw_16 = (real'(ea_16b) / 65536.0) * (real'(man_16b) / 16.0);
            
            // Calculate Absolute Errors
            err_4  = expected_real - hw_4;  if (err_4 < 0)  err_4 = -err_4;
            err_8  = expected_real - hw_8;  if (err_8 < 0)  err_8 = -err_8;
            err_16 = expected_real - hw_16; if (err_16 < 0) err_16 = -err_16;
            
            if (err_4 > max_err_4)   max_err_4 = err_4;
            if (err_8 > max_err_8)   max_err_8 = err_8;
            if (err_16 > max_err_16) max_err_16 = err_16;

            sum_err_4 += err_4;
            sum_err_8 += err_8;
            sum_err_16 += err_16;
            
            tests_run++;
        end

        $display(" [4-BIT LUT]  Max Error : %f | Avg Error : %f", max_err_4, sum_err_4 / tests_run);
        $display(" [8-BIT LUT]  Max Error : %f | Avg Error : %f", max_err_8, sum_err_8 / tests_run);
        $display(" [16-BIT LUT] Max Error : %f | Avg Error : %f", max_err_16, sum_err_16 / tests_run);
        $display("=======================================================================\n");
        $finish;
    end
endmodule