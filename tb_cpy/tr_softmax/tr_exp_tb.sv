`timescale 1ns/1ps

module tr_exp_tb();

    // ========================================================================
    // 8-BIT DUT INSTANTIATIONS (SoftMax / GELU - Q4.4)
    // ========================================================================
    logic signed [7:0] x_8b;
    logic [7:0] ea_8b_0, man_8b_0; logic z_8b_0;
    logic [7:0] ea_8b_1, man_8b_1; logic z_8b_1;
    logic [7:0] ea_8b_2, man_8b_2; logic z_8b_2;
    
    // Using the combined module in 8-bit mode
    tr_exp #(.WIDTH(8), .FRAC_W(4), .LUT_IDX_W(3), .ITER(0)) u_8b_0 (.x(x_8b), .e_a(ea_8b_0), .mantisa(man_8b_0), .is_zero(z_8b_0));
    tr_exp #(.WIDTH(8), .FRAC_W(4), .LUT_IDX_W(3), .ITER(1)) u_8b_1 (.x(x_8b), .e_a(ea_8b_1), .mantisa(man_8b_1), .is_zero(z_8b_1));
    tr_exp #(.WIDTH(8), .FRAC_W(4), .LUT_IDX_W(3), .ITER(2)) u_8b_2 (.x(x_8b), .e_a(ea_8b_2), .mantisa(man_8b_2), .is_zero(z_8b_2));

    // ========================================================================
    // 20-BIT DUT INSTANTIATIONS (LayerNorm - Q12.8)
    // ========================================================================
    logic signed [19:0] x_20b;
    logic [19:0] ea_20b_0, man_20b_0; logic z_20b_0;
    logic [19:0] ea_20b_1, man_20b_1; logic z_20b_1;
    logic [19:0] ea_20b_2, man_20b_2; logic z_20b_2;

    // Using the new dedicated Q12.8 module
    q12_8_tr_exp #(.ITER(0)) u_20b_0 (.x(x_20b), .e_a(ea_20b_0), .mantisa(man_20b_0), .is_zero(z_20b_0));
    q12_8_tr_exp #(.ITER(1)) u_20b_1 (.x(x_20b), .e_a(ea_20b_1), .mantisa(man_20b_1), .is_zero(z_20b_1));
    q12_8_tr_exp #(.ITER(2)) u_20b_2 (.x(x_20b), .e_a(ea_20b_2), .mantisa(man_20b_2), .is_zero(z_20b_2));

    // ========================================================================
    // VERIFICATION VARIABLES
    // ========================================================================
    real x_real, expected_real;
    real hw_0, hw_1, hw_2;
    real err_0, err_1, err_2;
    
    real max_err_8b_0 = 0.0, max_err_8b_1 = 0.0, max_err_8b_2 = 0.0;
    real max_err_20b_0 = 0.0, max_err_20b_1 = 0.0, max_err_20b_2 = 0.0;

    int tests_run_8b = 0, tests_run_20b = 0;

    initial begin
        $display("=======================================================================");
        $display(" STARTING MULTI-ORDER TR-EXP VERIFICATION");
        $display("=======================================================================");

        // --------------------------------------------------------------------
        // SWEEP 1: 8-BIT MODE (Strictly Negative Domain: -128 to 0)
        // --------------------------------------------------------------------
        for (int i = -128; i <= 0; i++) begin
            x_8b = i;
            #1;
            
            x_real = real'(i) / 16.0;
            expected_real = $exp(x_real);
            
            // Reconstruct Hardware Outputs
            hw_0 = real'(ea_8b_0) / 256.0;                                        // 0-Order
            hw_1 = (real'(ea_8b_1) / 256.0) * (real'(man_8b_1) / 16.0);           // 1st-Order
            hw_2 = (real'(ea_8b_2) / 256.0) * (real'(man_8b_2) / 16.0);           // 2nd-Order
            
            // Calculate Absolute Errors
            err_0 = expected_real - hw_0; if (err_0 < 0) err_0 = -err_0;
            err_1 = expected_real - hw_1; if (err_1 < 0) err_1 = -err_1;
            err_2 = expected_real - hw_2; if (err_2 < 0) err_2 = -err_2;
            
            if (err_0 > max_err_8b_0) max_err_8b_0 = err_0;
            if (err_1 > max_err_8b_1) max_err_8b_1 = err_1;
            if (err_2 > max_err_8b_2) max_err_8b_2 = err_2;
            
            tests_run_8b++;
        end

        // --------------------------------------------------------------------
        // SWEEP 2: 20-BIT MODE (LayerNorm Domain: -1536 to +768)
        // --------------------------------------------------------------------
        // Range covers approx -6.0 to +3.0 in Q12.8 format (1.0 = 256)
        for (int i = -1536; i <= 768; i++) begin
            x_20b = i;
            #1;
            
            x_real = real'(i) / 256.0;
            expected_real = $exp(x_real);
            
            // Reconstruct Hardware Outputs (Mantissa is Qx.8 here, Anchor is Qx.8)
            hw_0 = real'(ea_20b_0) / 256.0;                                        // 0-Order
            hw_1 = (real'(ea_20b_1) / 256.0) * (real'(man_20b_1) / 256.0);         // 1st-Order
            hw_2 = (real'(ea_20b_2) / 256.0) * (real'(man_20b_2) / 256.0);         // 2nd-Order
            
            // Calculate Absolute Errors
            err_0 = expected_real - hw_0; if (err_0 < 0) err_0 = -err_0;
            err_1 = expected_real - hw_1; if (err_1 < 0) err_1 = -err_1;
            err_2 = expected_real - hw_2; if (err_2 < 0) err_2 = -err_2;
            
            if (x_real <= 2.5) begin
                if (err_0 > max_err_20b_0) max_err_20b_0 = err_0;
                if (err_1 > max_err_20b_1) max_err_20b_1 = err_1;
                if (err_2 > max_err_20b_2) max_err_20b_2 = err_2;
            end
            
            tests_run_20b++;
        end

        // --------------------------------------------------------------------
        // FINAL VERIFICATION REPORT
        // --------------------------------------------------------------------
        $display("\n=======================================================================");
        $display(" VERIFICATION REPORT: TR-EXP ERROR CONVERGENCE");
        $display("=======================================================================");
        
        $display(" [8-BIT MODE : SoftMax/GELU (Q4.4) | %0d vectors tested]", tests_run_8b);
        $display("    -> 0-Order Max Error : %f", max_err_8b_0);
        $display("    -> 1-Order Max Error : %f", max_err_8b_1);
        $display("    -> 2-Order Max Error : %f", max_err_8b_2);
        
        $display("\n [20-BIT MODE : LayerNorm (Q12.8) | %0d vectors tested]", tests_run_20b);
        $display("    -> 0-Order Max Error : %f", max_err_20b_0);
        $display("    -> 1-Order Max Error : %f", max_err_20b_1);
        $display("    -> 2-Order Max Error : %f", max_err_20b_2);
        $display("=======================================================================\n");
        $finish;
    end
    
endmodule