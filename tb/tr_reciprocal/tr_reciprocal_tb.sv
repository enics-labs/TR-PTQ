`timescale 1ns/1ps

module tr_reciprocal_tb;

    // ========================================================================
    // MODE 1: SOFTMAX (Reciprocal: 1/x)
    // ========================================================================
    logic [15:0] x_sm;
    logic [7:0]  y_sm;
    
    tr_reciprocal #(
        .IN_WIDTH(16), .OUT_WIDTH(8), 
        .IN_FRAC(4), .OUT_FRAC(4), 
        .INV_SQRT(0), .ITER(2)
    ) u_softmax_div (
        .clk(1'b0), .rst_n(1'b1),
        .xq(x_sm), .yq(y_sm)
    );

    // ========================================================================
    // MODE 2: LAYERNORM (Inverse Square Root: 1/sqrt(x))
    // ========================================================================
    logic [19:0] x_ln;
    logic [11:0] y_ln;
    
    tr_reciprocal #(
        .IN_WIDTH(20), .OUT_WIDTH(12), 
        .IN_FRAC(8), .OUT_FRAC(8), 
        .INV_SQRT(1), .ITER(2)
    ) u_layernorm_isqrt (
        .clk(1'b0), .rst_n(1'b1),
        .xq(x_ln), .yq(y_ln)
    );

    real x_real, expected_real, hw_real, current_error;
    real max_err_sm = 0.0;
    real max_err_ln = 0.0;

    initial begin
        $display("=======================================================================");
        $display(" STARTING TR-RECIPROCAL VERIFICATION (Reciprocal & Inv-Sqrt)");
        $display("=======================================================================");

        // --------------------------------------------------------------------
        // SWEEP 1: SOFTMAX DENOMINATOR (1/x)
        // --------------------------------------------------------------------
        // Sweeping sum of exponentials from 1.0 (16) up to 15.0 (240)
        for (int i = 16; i <= 240; i++) begin
            x_sm = i; #1;
            
            hw_real       = real'(y_sm) / 256.0;
            x_real        = real'(i) / 16.0;
            expected_real = 1.0 / x_real;
            
            current_error = expected_real - hw_real;
            if (current_error < 0) current_error = -current_error;
            if (current_error > max_err_sm) max_err_sm = current_error;
        end

        // --------------------------------------------------------------------
        // SWEEP 2: LAYERNORM VARIANCE (1/sqrt(x))
        // --------------------------------------------------------------------
        // Sweeping variance from 0.25 (64) up to 3.0 (768)
        for (int i = 64; i <= 768; i++) begin
            x_ln = i; #1;
            
            hw_real       = real'(y_ln) / 256.0;
            x_real        = real'(i) / 256.0;
            // Native SV doesn't have a fast $sqrt, so we use $pow
            expected_real = 1.0 / ($pow(x_real, 0.5)); 
            
            current_error = expected_real - hw_real;
            if (current_error < 0) current_error = -current_error;
            if (current_error > max_err_ln) max_err_ln = current_error;
        end

        // --------------------------------------------------------------------
        // FINAL VERIFICATION REPORT
        // --------------------------------------------------------------------
        $display("\n=======================================================================");
        $display(" VERIFICATION REPORT: TR-RECIPROCAL");
        $display("=======================================================================");
        $display(" [SOFTMAX MODE : 1/x]");
        $display("    -> Max Error  : %f", max_err_sm);
        
        $display("\n [LAYERNORM MODE : 1/sqrt(x)]");
        $display("    -> Max Error  : %f", max_err_ln);
        $display("=======================================================================\n");
        $finish;
    end

endmodule