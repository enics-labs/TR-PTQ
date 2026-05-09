`timescale 1ns/1ps

module tb_tr_gelu();

    localparam int N = 8;
    localparam int W = 8;
    localparam int ACC_W = 32;
    localparam real SCALE_4 = 16.0;   

    logic                 clk, rst_n, valid_in;
    logic [1:0]           mode;       
    logic signed [W-1:0]  x_in [N];
    logic signed [W-1:0]  aux_in [N];
    
    logic                 valid_out;
    logic signed [W-1:0]  y_out [N];

    tr_gelu #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) dut (.*);

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 0; valid_in = 0; mode = 2'b00;
        for (int i=0; i<N; i++) begin x_in[i] = '0; aux_in[i] = '0; end
        #22 rst_n = 1;

        $display("=======================================================================");
        $display(" FULL STATELESS TR-GELU VERIFICATION (WITH INTERMEDIATE LOGS)");
        $display("=======================================================================");

        // Test Vector: [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, 3.0]
        test_stateless_gelu('{32, 16, 8, 0, -8, -16, -32, 48});
        
        $finish;
    end

    task automatic test_stateless_gelu(input logic signed [W-1:0] vec [N]);
        
        // Controller SRAM Vectors
        logic signed [W-1:0]  ctrl_E [N];
        logic signed [W-1:0]  ctrl_recip [N];
        
        real x_float, exp_E, exp_recip, exp_sig, exp_y, hw_y, err;

        $display("\n---> Testing Vector: %p", vec);

        // =========================================================
        // PASS 1: Alpha & Exponential Generator
        // =========================================================
        @(negedge clk);
        mode = 2'b00; x_in = vec; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        
        $display("\n   [PASS 1] Calculated Exponential Vector (E = e^(-a|x|)):");
        for(int i=0; i<N; i++) begin 
            ctrl_E[i] = y_out[i]; 
            $display("      Idx %0d | HW E: %6.3f", i, real'(ctrl_E[i]) / SCALE_4);
        end
        repeat(3) @(posedge clk);

        // =========================================================
        // PASS 2: Denominator & Reciprocal
        // =========================================================
        @(negedge clk);
        mode = 2'b01; x_in = ctrl_E; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        
        $display("\n   [PASS 2] Calculated Reciprocal Vector 1 / (1+E):");
        for(int i=0; i<N; i++) begin 
            ctrl_recip[i] = y_out[i]; 
            $display("      Idx %0d | HW Recip: %6.3f", i, real'(ctrl_recip[i]) / SCALE_4);
        end
        repeat(3) @(posedge clk);

        // =========================================================
        // PASS 3: Symmetry & Final Product
        // =========================================================
        @(negedge clk);
        mode = 2'b10; x_in = vec; aux_in = ctrl_recip; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        
        $display("\n   [PASS 3] Final GELU Output Comparison:");
        for (int i=0; i<N; i++) begin
            x_float = real'(vec[i]) / SCALE_4;
            
            // GELU Float Math using Alpha Approximation
            exp_E     = $exp(-1.702 * ((x_float > 0) ? x_float : -x_float));
            exp_recip = 1.0 / (1.0 + exp_E);
            exp_sig   = (x_float < 0) ? (1.0 - exp_recip) : exp_recip;
            exp_y     = x_float * exp_sig;
            
            hw_y = real'($signed(y_out[i])) / SCALE_4;
            err  = (hw_y > exp_y) ? (hw_y - exp_y) : (exp_y - hw_y);
            
            if (err < 0.15)
                $display("      \033[0;32m[OK]\033[0m Idx %0d | X: %5.1f | Exp GELU: %6.3f | HW GELU: %6.3f", i, x_float, exp_y, hw_y);
            else
                $display("      \033[0;31m[FAIL]\033[0m Idx %0d | X: %5.1f | Exp GELU: %6.3f | HW GELU: %6.3f", i, x_float, exp_y, hw_y);
        end
        
        repeat(5) @(posedge clk);
    endtask

endmodule