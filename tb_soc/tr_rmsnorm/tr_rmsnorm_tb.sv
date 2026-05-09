`timescale 1ns/1ps

module tb_tr_rmsnorm();

    localparam int N = 8;
    localparam int W = 8;
    localparam int ACC_W = 32;
    localparam real SCALE_4 = 16.0;   
    localparam real SCALE_8 = 256.0;

    // Controller Constant: 0.5 * ln(N). For N=8, ln(sqrt(8)) = ~1.0397. 
    // In Q4.4 hardware format: 1.0397 * 16 = 16.635 => 17
    localparam logic signed [W-1:0] CONST_LN_SQRT_N = 8'd17; 

    logic                 clk, rst_n, valid_in;
    logic [1:0]           mode;       
    logic signed [W-1:0]  x_in [N];
    logic signed [W-1:0]  aux_in [N];
    logic signed [ACC_W-1:0] sum_in;
    logic signed [W-1:0]  offset_in;
    
    logic                 valid_out;
    logic signed [W-1:0]  y_out [N];
    logic signed [ACC_W-1:0] sum_out;
    logic signed [W-1:0]  ln_out;

    tr_rmsnorm #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) dut (.*);

    initial begin
        clk = 0; forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 0; valid_in = 0; mode = 2'b00; sum_in = '0; offset_in = '0;
        for (int i=0; i<N; i++) begin x_in[i] = '0; aux_in[i] = '0; end
        #22 rst_n = 1;

        $display("=======================================================================");
        $display(" FULL STATELESS TR-RMSNORM VERIFICATION");
        $display("=======================================================================");

        // Test Vector: [2.0, 1.0, 0.0, -1.0, -2.0, 0.0, 1.0, -1.0]
        // Gamma Vector: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] (for simplicity)
        test_rmsnorm(
            '{32, 16, 0, -16, -32, 0, 16, -16}, 
            '{16, 16, 16, 16, 16, 16, 16, 16}
        );
        
        $finish;
    end

    task automatic test_rmsnorm(input logic signed [W-1:0] vec [N], input logic signed [W-1:0] gamma [N]);
        
        // Controller Memory
        logic signed [ACC_W-1:0] ctrl_sum;
        logic signed [W-1:0]     ctrl_V [N]; // X * Gamma
        logic signed [W-1:0]     ctrl_log_offset;
        logic signed [W-1:0]     ctrl_inv_rms;
        
        real x_float, exp_sum, exp_rms, exp_inv_rms, exp_y, hw_y, err;

        $display("\n---> Testing Vector: %p", vec);

        // =========================================================
        // PASS 1: SUM OF SQUARES (DOT PRODUCT)
        // =========================================================
        @(negedge clk);
        mode = 2'b00; x_in = vec; aux_in = vec; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        ctrl_sum = sum_out; 
        $display("   [PASS 1] Sum of Squares: %6.3f", real'(ctrl_sum) / SCALE_8);
        repeat(3) @(posedge clk);

        // =========================================================
        // PASS 2: LOGARITHM & X*GAMMA
        // =========================================================
        @(negedge clk);
        mode = 2'b01; x_in = vec; aux_in = gamma; sum_in = ctrl_sum; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        for(int i=0; i<N; i++) ctrl_V[i] = y_out[i];
        
        // Controller Math: Add the constant!
        ctrl_log_offset = ln_out + CONST_LN_SQRT_N;
        
        $display("   [PASS 2] -0.5 * ln(S): %6.3f", real'($signed(ln_out)) / SCALE_4);
        $display("   [CTRL]   LogOffset (-0.5*ln(S) + ln(sqrt(N))): %6.3f", real'($signed(ctrl_log_offset)) / SCALE_4);
        repeat(3) @(posedge clk);

        // =========================================================
        // PASS 3: INVERSE RMS RECONSTRUCTION
        // =========================================================
        @(negedge clk);
        mode = 2'b10; offset_in = ctrl_log_offset; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        ctrl_inv_rms = y_out[0];
        $display("   [PASS 3] Scalar InvRMS (e^LogOffset): %6.3f", real'($signed(ctrl_inv_rms)) / SCALE_4);
        repeat(3) @(posedge clk);

        // =========================================================
        // PASS 4: FINAL SCALING (V * InvRMS)
        // =========================================================
        @(negedge clk);
        mode = 2'b11; x_in = ctrl_V; 
        for(int i=0; i<N; i++) aux_in[i] = ctrl_inv_rms; // Broadcast scalar
        valid_in = 1'b1;
        
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        
        $display("\n   [PASS 4] Final RMSNorm Output:");
        
        // Golden Model Math
        exp_sum = 0.0;
        for(int i=0; i<N; i++) exp_sum += (real'(vec[i])/SCALE_4) * (real'(vec[i])/SCALE_4);
        exp_rms = $sqrt(exp_sum / N);
        exp_inv_rms = 1.0 / exp_rms;

        for (int i=0; i<N; i++) begin
            x_float = real'(vec[i]) / SCALE_4;
            exp_y   = (x_float * exp_inv_rms) * (real'(gamma[i]) / SCALE_4);
            
            hw_y = real'($signed(y_out[i])) / SCALE_4;
            err  = (hw_y > exp_y) ? (hw_y - exp_y) : (exp_y - hw_y);
            
            if (err < 0.15)
                $display("      \033[0;32m[OK]\033[0m X: %5.1f | Exp Y: %6.3f | HW Y: %6.3f", x_float, exp_y, hw_y);
            else
                $display("      \033[0;31m[FAIL]\033[0m X: %5.1f | Exp Y: %6.3f | HW Y: %6.3f", x_float, exp_y, hw_y);
        end
        
        repeat(5) @(posedge clk);
    endtask

endmodule