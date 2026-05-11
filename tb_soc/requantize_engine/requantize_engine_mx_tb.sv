`timescale 1ns/1ps

module requantize_engine_mx_tb();

    // Parameters
    localparam int N = 4;
    localparam int ACC_W = 32;
    localparam int OUT_W = 8;

    // DUT Signals
    logic               clk, rst_n;
    logic               dot_in_valid, req_out_valid;
    logic signed [ACC_W-1:0] dot_in [N];
    logic signed [7:0]       exp_act_in, exp_weight_in;
    logic signed [OUT_W-1:0] req_vec_out [N];
    logic signed [7:0]       exp_total_out;

    // DUT Instantiation
    requantize_engine_mx #(
        .N(N),
        .ACC_W(ACC_W),
        .OUT_W(OUT_W)
    ) dut (.*);

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("==================================================");
        $display("  Starting Corrected MX Requantizer Tests...      ");
        $display("==================================================");

        // System Reset
        rst_n = 0;
        dot_in_valid = 0;
        exp_act_in = 0;
        exp_weight_in = 0;
        for (int i=0; i<N; i++) dot_in[i] = 0;
        
        #20 rst_n = 1;
        @(posedge clk);

        // ====================================================================
        // TEST 1: No Compression Needed (Values fit in 8-bit)
        // dot_in = {100, -50, 25, 0}
        // E_act = 2, E_wgt = 3  =>  Base Exponent = 5
        // Max Abs = 100 (Fits in 7 magnitude bits). Shift Needed = 0.
        // Expected Mantissas: {100, -50, 25, 0}
        // Expected Exponent: 5 + 0 = 5
        // ====================================================================
        $display("\n[TEST 1] No Compression (Values fit securely in 8-bit)");
        dot_in_valid  = 1;
        exp_act_in    = 8'sd2;
        exp_weight_in = 8'sd3;
        dot_in        = '{32'd100, -32'sd50, 32'd25, 32'd0};
        
        @(posedge clk);
        dot_in_valid = 0;
        @(posedge clk); 
        
        if (req_vec_out[0] === 8'd100 && req_vec_out[1] === -8'sd50 && 
            req_vec_out[2] === 8'd25 && req_vec_out[3] === 8'd0 &&
            exp_total_out === 8'sd5)
            $display("  -> [PASS] Mantissas untouched, exponent calculated correctly.");
        else
            $error("  -> [FAIL] Test 1 Incorrect.");

        // ====================================================================
        // TEST 2: Dynamic Block Compression (Max value > 127)
        // dot_in = {1000, -2000, 500, 0}
        // E_act = 0, E_wgt = 0  =>  Base Exponent = 0
        // Max Abs = 2000. 2000 is 11 bits wide (11111010000).
        // To fit in 7 magnitude bits, we must right-shift by 4.
        // Expected Exponent: Base(0) + Shift(4) = 4
        // Expected Mantissas: 1000>>4=62, -2000>>4=-125, 500>>4=31
        // ====================================================================
        $display("\n[TEST 2] Dynamic Block Compression (Requires Shift)");
        @(posedge clk);
        dot_in_valid  = 1;
        exp_act_in    = 8'sd0;
        exp_weight_in = 8'sd0;
        dot_in        = '{32'd1000, -32'sd2000, 32'd500, 32'd0};
        
        @(posedge clk);
        dot_in_valid = 0;
        @(posedge clk);
        
        if (req_vec_out[0] === 8'd62 && req_vec_out[1] === -8'sd125 && 
            req_vec_out[2] === 8'd31 && req_vec_out[3] === 8'd0 &&
            exp_total_out === 8'sd4)
            $display("  -> [PASS] Block accurately compressed and new exponent generated.");
        else
            $error("  -> [FAIL] Compression logic failed.");

        // ====================================================================
        // TEST 3: Heavy Scaling 
        // dot_in = {65536, -32768, 16384, 8192}
        // Max Abs = 65536. 65536 is 17 bits wide. Shift Needed = 10.
        // E_act = -2, E_wgt = -1 => Base Exponent = -3
        // Expected Exponent = -3 + 10 = 7
        // Expected Mantissas: 65536>>10=64, -32768>>10=-32, 16384>>10=16, 8192>>10=8
        // ====================================================================
        $display("\n[TEST 3] Heavy Scaling (Massive Accumulators)");
        @(posedge clk);
        dot_in_valid  = 1;
        exp_act_in    = -8'sd2;
        exp_weight_in = -8'sd1;
        dot_in        = '{32'd65536, -32'sd32768, 32'd16384, 32'd8192};
        
        @(posedge clk);
        dot_in_valid = 0;
        @(posedge clk);

        if (req_vec_out[0] === 8'd64 && req_vec_out[1] === -8'sd32 && 
            req_vec_out[2] === 8'd16 && req_vec_out[3] === 8'd8 &&
            exp_total_out === 8'sd7)
            $display("  -> [PASS] Massive block accurately scaled down to MXINT8.");
        else
            $error("  -> [FAIL] Heavy scaling failed.");

        $display("\n==================================================");
        $display("  ALL MX REQUANTIZER TESTS COMPLETED.             ");
        $display("==================================================");
        $finish;
    end
endmodule