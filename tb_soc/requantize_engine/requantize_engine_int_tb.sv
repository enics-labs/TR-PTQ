`timescale 1ns/1ps

module requantize_engine_int_tb();

    localparam int N       = 4;
    localparam int ACC_W   = 32;
    localparam int MUL_W   = 32;
    localparam int SHIFT_W = 6;
    localparam int OUT_W   = 8;

    logic clk, rst_n;
    logic in_valid, in_ready;
    logic signed [ACC_W-1:0] acc_in [N];
    logic signed [MUL_W-1:0] multiplier;
    logic [SHIFT_W-1:0]      shift;
    
    logic out_valid, out_ready;
    logic signed [OUT_W-1:0] out_vec [N];

    // DUT
    requantize_engine_int #(
        .N(N), .ACC_W(ACC_W), .MUL_W(MUL_W), .SHIFT_W(SHIFT_W), .OUT_W(OUT_W)
    ) dut (.*);

    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("Starting Requantization Tests...");
        rst_n = 0;
        in_valid = 0;
        out_ready = 1;
        
        #20 rst_n = 1;
        @(posedge clk);

        // =========================================================
        // Test Case: Scale = 0.00390625 (1/256)
        // Multiplier = 1, Shift = 8
        // Logic: (Acc * 1 + 128) >>> 8
        // =========================================================
        multiplier = 32'd1;
        shift = 6'd8;
        
        // Data Setup:
        // L0: 25600 -> 25600 / 256 = 100 (Clean scale)
        // L1: 3968  -> 3968 / 256 = 15.5 -> Rounds up to 16 (Rounding test)
        // L2: 80000 -> 80000 / 256 = 312.5 -> Clamps to 127 (Positive Outlier)
        // L3: -90000-> -90000 / 256 = -351.5 -> Clamps to -128 (Negative Outlier)
        
        acc_in = '{32'd25600, 32'd3968, 32'd80000, -32'sd90000};
        
        in_valid = 1;
        wait(in_ready);
        @(posedge clk);
        in_valid = 0;

        // Wait 3 cycles for pipeline
        wait(out_valid);
        @(posedge clk);

        $display("Results (Multiplier: %0d, Shift: %0d)", multiplier, shift);
        $display("Lane 0 (Clean)    | Expected: 100  | Got: %0d", out_vec[0]);
        $display("Lane 1 (Rounding) | Expected: 16   | Got: %0d", out_vec[1]);
        $display("Lane 2 (Sat +)    | Expected: 127  | Got: %0d", out_vec[2]);
        $display("Lane 3 (Sat -)    | Expected: -128 | Got: %0d", out_vec[3]);

        if (out_vec[0] === 8'd100 && out_vec[1] === 8'd16 && 
            out_vec[2] === 8'd127 && out_vec[3] === -8'sd128) begin
            $display("\n[PASS] Requantization pipeline executed flawlessly!");
        end else begin
            $error("\n[FAIL] Requantization logic mismatch.");
        end

        #20 $finish;
    end

endmodule