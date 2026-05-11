`timescale 1ns/1ps

module formatter_mx_tb();

    localparam int N = 4;
    localparam int VPU_W = 16;
    localparam int MX_W = 8;

    // DUT Signals
    logic                 clk, rst_n, valid_in, valid_out;
    logic signed [VPU_W-1:0] vpu_data_in [N];
    logic signed [MX_W-1:0]  mx_mantissas [N];
    logic signed [7:0]    mx_shared_exp;

    formatter_mx #(
        .N(N), .VPU_W(VPU_W), .MX_W(MX_W)
    ) dut (.*);

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        $display("==================================================");
        $display("  Starting MX Formatter Tests...                  ");
        $display("==================================================");

        rst_n = 0;
        valid_in = 0;
        for(int i=0; i<N; i++) vpu_data_in[i] = 0;
        #20 rst_n = 1;
        @(posedge clk);

        // ====================================================================
        // TEST 1: Small numbers (No shift required)
        // input = {10, -100, 127, 0}
        // max abs = 127. Fits in 7 bits of magnitude. E_out should be 0.
        // ====================================================================
        $display("\n[TEST 1] Small Numbers (E_out = 0)");
        valid_in = 1;
        vpu_data_in = '{16'sd10, -16'sd100, 16'sd127, 16'sd0};
        @(posedge clk);
        valid_in = 0;
        @(posedge clk); // Pipeline stage

        if (mx_shared_exp === 8'd0 && mx_mantissas[1] === -8'sd100)
            $display("  -> [PASS] Correctly bypassed shift for small numbers.");
        else
            $error("  -> [FAIL] Expected Exp 0, got %0d", mx_shared_exp);

        // ====================================================================
        // TEST 2: Large numbers (Requires scaling)
        // input = {1000, -2000, 42, 8}
        // max abs = 2000. 
        // 2000 in binary is 11111010000 (11 bits wide). 
        // Target is 7 bits wide. E_out = 11 - 7 = 4. 
        // Shift right by 4. -2000 >> 4 = -125. 1000 >> 4 = 62.
        // ====================================================================
        $display("\n[TEST 2] Large Numbers (Requires Compression)");
        valid_in = 1;
        vpu_data_in = '{16'sd1000, -16'sd2000, 16'sd42, 16'sd8};
        @(posedge clk);
        valid_in = 0;
        @(posedge clk);

        if (mx_shared_exp === 8'd4 && mx_mantissas[0] === 8'sd62 && mx_mantissas[1] === -8'sd125)
            $display("  -> [PASS] Exp calculated correctly (4) and shifted properly.");
        else
            $error("  -> [FAIL] Expected Exp 4, got %0d. M1: %0d", mx_shared_exp, mx_mantissas[1]);

        $display("\n==================================================");
        $finish;
    end
endmodule