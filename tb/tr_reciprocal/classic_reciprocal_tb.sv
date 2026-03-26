`timescale 1ns/1ps

module tr_reciprocal_tb;

    logic clk;
    logic rst_n;

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ========================================================================
    // MODE 1: CLASSIC DIVIDER (SoftMax: 1/x)
    // ========================================================================
    logic [15:0] x_sm;
    logic [7:0]  y_sm;
    logic        start_sm;
    logic        valid_sm;
    
    classic_reciprocal #(
        .IN_WIDTH(16), .OUT_WIDTH(8), 
        .IN_FRAC(4), .OUT_FRAC(4)
    ) u_classic_div (
        .clk(clk), .rst_n(rst_n),
        .start(start_sm),
        .xq(x_sm), 
        .valid_out(valid_sm),
        .yq(y_sm)
    );

    // ========================================================================
    // MODE 2: TR-RECIPROCAL (LayerNorm: 1/sqrt(x))
    // ========================================================================
    logic [19:0] x_ln;
    logic [11:0] y_ln;
    
    tr_reciprocal #(
        .IN_WIDTH(20), .OUT_WIDTH(12), 
        .IN_FRAC(8), .OUT_FRAC(8), 
        .INV_SQRT(1), .ITER(2)
    ) u_layernorm_isqrt (
        .clk(clk), .rst_n(rst_n),
        .xq(x_ln), .yq(y_ln)
    );

    real x_real, expected_real, hw_real, current_error;
    real max_err_sm;
    real max_err_ln;

    initial begin
        max_err_sm = 0.0;
        max_err_ln = 0.0;
        
        rst_n = 0;
        start_sm = 0;
        x_sm = 0;
        x_ln = 0;
        
        #20 rst_n = 1;
        
        $display("=======================================================================");
        $display(" STARTING RECIPROCAL VERIFICATION (Classic Divider & TR Inv-Sqrt)");
        $display("=======================================================================");

        // --------------------------------------------------------------------
        // SWEEP 1: SOFTMAX DENOMINATOR (Classic Divider: 1/x)
        // --------------------------------------------------------------------
        for (int i = 16; i <= 240; i++) begin
            @(posedge clk);
            #1; // FIX 1: Drive signals slightly AFTER the clock edge to avoid races!
            x_sm = i;
            start_sm = 1;
            
            @(posedge clk);
            #1;
            start_sm = 0;
            
            // FIX 2: Wait strictly for the transition from 0 to 1
            @(posedge valid_sm);
            
            // Extract the result on the valid cycle
            hw_real       = real'(y_sm) / 16.0; 
            x_real        = real'(i) / 16.0;    
            expected_real = 1.0 / x_real;
            
            current_error = expected_real - hw_real;
            if (current_error < 0) current_error = -current_error;
            if (current_error > max_err_sm) max_err_sm = current_error;
        end

        // --------------------------------------------------------------------
        // SWEEP 2: LAYERNORM VARIANCE (TR-Reciprocal: 1/sqrt(x))
        // --------------------------------------------------------------------
        for (int i = 64; i <= 768; i++) begin
            @(posedge clk);
            #1; // Align to active region
            x_ln = i; 
            #1; // Allow combinational logic to evaluate
            
            hw_real       = real'(y_ln) / 256.0; 
            x_real        = real'(i) / 256.0;    
            
            expected_real = 1.0 / ($pow(x_real, 0.5)); 
            
            current_error = expected_real - hw_real;
            if (current_error < 0) current_error = -current_error;
            if (current_error > max_err_ln) max_err_ln = current_error;
        end

        // --------------------------------------------------------------------
        // FINAL VERIFICATION REPORT
        // --------------------------------------------------------------------
        $display("\n=======================================================================");
        $display(" VERIFICATION REPORT: RECIPROCAL ENGINES");
        $display("=======================================================================");
        $display(" [CLASSIC DIVIDER MODE : 1/x]");
        $display("    -> Max Error  : %f", max_err_sm);
        
        $display("\n [TR-LAYERNORM MODE : 1/sqrt(x)]");
        $display("    -> Max Error  : %f", max_err_ln);
        $display("=======================================================================\n");
        $finish;
    end

endmodule