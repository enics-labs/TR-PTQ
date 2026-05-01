`timescale 1ns/1ps

module mac_array_engine_tb();

    // Parameters
    localparam int N = 4; // Kept small for easy console reading
    localparam int W = 8;
    localparam int ACC_W = 32;

    // Signals
    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [1:0] op_mode;
    logic [W-1:0] a [N], b [N];
    logic signed [ACC_W-1:0] c;
    logic clear_acc;
    
    logic out_valid, out_ready;
    logic signed [ACC_W-1:0] out_dot;

    // DUT Instantiation
    mac_array_engine #(
        .N(N), .W(W), .ACC_W(ACC_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready),
        .op_mode(op_mode), .a(a), .b(b), .c(c), .clear_acc(clear_acc),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_dot(out_dot)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Task: Feed Data and Wait for Result
    task run_mac_test(
        input logic [1:0] mode,
        input logic clr,
        input logic signed [ACC_W-1:0] bias,
        input int val_a [N],
        input int val_b [N],
        input string test_name
    );
        logic signed [ACC_W-1:0] expected_sum;
        int i;

        // 1. Calculate Expected Result Behaviorally
        expected_sum = clr ? bias : out_dot; // If clearing, start with bias. Else, accumulate.
        for (i = 0; i < N; i++) begin
            if (mode == 2'd0)      expected_sum += $signed(val_a[i]) * $signed(val_b[i]); // SS
            else if (mode == 2'd1) expected_sum += $signed(val_a[i]) * $signed({1'b0, val_b[i][W-1:0]}); // SU
            else                   expected_sum += $signed({1'b0, val_a[i][W-1:0]}) * $signed({1'b0, val_b[i][W-1:0]}); // UU
        end

        // 2. Drive Inputs
        @(posedge clk);
        in_valid <= 1;
        op_mode <= mode;
        clear_acc <= clr;
        c <= bias;
        for (i = 0; i < N; i++) begin
            a[i] <= val_a[i];
            b[i] <= val_b[i];
        end

        // Wait for acceptance
        wait(in_ready);
        @(posedge clk);
        in_valid <= 0;

        // 3. Wait for Output Pipeline
        wait(out_valid);
        @(posedge clk);

        // 4. Check Assertion
        if (out_dot === expected_sum) begin
            $display("[PASS] %s | Expected: %0d, Got: %0d", test_name, expected_sum, out_dot);
        end else begin
            $error("[FAIL] %s | Expected: %0d, Got: %0d", test_name, expected_sum, out_dot);
            $finish;
        end
    endtask

    // Main Test Sequence
    initial begin
        $display("Starting mac_array_engine Tests...");
        rst_n = 0;
        in_valid = 0;
        out_ready = 1; // Always ready to receive
        
        #20 rst_n = 1;
        @(posedge clk);

        // Test 1: Basic Signed * Signed (Clear Acc, Zero Bias)
        // a = {2, -3, 4, -5}, b = {10, 10, 10, 10} -> Sum: 20 - 30 + 40 - 50 = -20
        run_mac_test(2'd0, 1, 0, '{2, -3, 4, -5}, '{10, 10, 10, 10}, "SS Basic Dot Product");

        // Test 2: Accumulation Mode (Don't clear, add to previous -20)
        // a = {1, 1, 1, 1}, b = {5, 5, 5, 5} -> Sum: 20. Total: -20 + 20 = 0
        run_mac_test(2'd0, 0, 0, '{1, 1, 1, 1}, '{5, 5, 5, 5}, "SS Accumulation");

        // Test 3: Affine Mode (+C Bias)
        // a = {2, 2, 2, 2}, b = {3, 3, 3, 3} -> Sum: 24. Bias: 100. Total: 124
        run_mac_test(2'd0, 1, 100, '{2, 2, 2, 2}, '{3, 3, 3, 3}, "SS Affine with +C");

        // Test 4: Signed * Unsigned (GELU gating emulation)
        // a = {-10, -10, 0, 0}, b = {255, 255, 0, 0} (255 is unsigned max) -> Sum: -2550 - 2550 = -5100
        run_mac_test(2'd1, 1, 0, '{-10, -10, 0, 0}, '{255, 255, 0, 0}, "SU Mixed Sign");

        // Test 5: Unsigned * Unsigned (Max Value bounds check)
        // a = {255, 255, 255, 255}, b = {2, 2, 2, 2} -> Sum: 510 * 4 = 2040
        run_mac_test(2'd2, 1, 0, '{255, 255, 255, 255}, '{2, 2, 2, 2}, "UU Bounds Test");

        #50 $display("ALL MAC TESTS PASSED!");
        $finish;
    end
endmodule