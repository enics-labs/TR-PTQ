`timescale 1ns/1ps

module tb_tr_softmax();

    localparam int N = 8;
    localparam int W = 8;
    localparam int ACC_W = 32;
    localparam real SCALE_4 = 16.0;   
    localparam real SCALE_12 = 4096.0;

    logic                 clk, rst_n, valid_in;
    logic [1:0]           mode;       
    logic signed [W-1:0]  x_in [N];
    logic signed [W-1:0]  offset_in;
    logic signed [ACC_W-1:0] sum_in;
    
    logic                 valid_out;
    logic signed [W-1:0]  y_out [N];
    logic signed [ACC_W-1:0] sum_out;

    tr_softmax #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) dut (.*);

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 0; valid_in = 0; mode = 2'b00; offset_in = '0; sum_in = '0;
        for (int i=0; i<N; i++) x_in[i] = '0;
        #22 rst_n = 1;

        $display("=======================================================================");
        $display(" FULL STATELESS SOFTMAX VERIFICATION (CONTROLLER EMULATION)");
        $display("=======================================================================");

        test_stateless_softmax('{32, 16, 0, -16, 48, 16, 32, 0});
        
        $finish;
    end

    task automatic test_stateless_softmax(input logic signed [W-1:0] vec [N]);
        
        // Controller's External Registers
        logic signed [W-1:0]  ctrl_max;
        logic signed [ACC_W-1:0] ctrl_sum;
        logic signed [W-1:0]  ctrl_ln_S;
        logic signed [W-1:0]  ctrl_combined_offset;

        $display("\n---> Testing Vector: %p", vec);

        // =========================================================
        // PASS 1: MAX EXTRACTION
        // =========================================================
        @(negedge clk);
        mode = 2'b00; x_in = vec; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        ctrl_max = y_out[0]; 
        $display("   [PASS 1] Controller saved Max: %0d", ctrl_max);
        repeat(3) @(posedge clk);

        // =========================================================
        // PASS 2: EXPONENTIAL SUMMATION
        // =========================================================
        @(negedge clk);
        mode = 2'b01; x_in = vec; offset_in = ctrl_max; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        ctrl_sum = sum_out;
        $display("   [PASS 2] Controller saved Sum: %6.3f", real'(ctrl_sum) / SCALE_12);
        repeat(3) @(posedge clk);

        // =========================================================
        // PASS 3: LOGARITHM 
        // =========================================================
        @(negedge clk);
        mode = 2'b10; sum_in = ctrl_sum; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        ctrl_ln_S = y_out[0];
        $display("   [PASS 3] Controller saved ln(S): %6.3f", real'(ctrl_ln_S) / SCALE_4);
        repeat(3) @(posedge clk);

        // =========================================================
        // CONTROLLER MATH: Calculate Combined Offset
        // =========================================================
        ctrl_combined_offset = ctrl_max + ctrl_ln_S;
        $display("   [CTRL] Calculated New Offset (m + ln(S)): %0d", ctrl_combined_offset);

        // =========================================================
        // PASS 4: FINAL PROBABILITIES
        // =========================================================
        @(negedge clk);
        mode = 2'b11; x_in = vec; offset_in = ctrl_combined_offset; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        
        do begin @(posedge clk); end while (!valid_out);
        
        $display("   [PASS 4] Final Hardware Probabilities:");
        for (int i=0; i<N; i++) begin
            $display("      Idx %0d: %6.3f", i, real'($signed(y_out[i])) / SCALE_4);
        end
        
        repeat(5) @(posedge clk);
    endtask

endmodule