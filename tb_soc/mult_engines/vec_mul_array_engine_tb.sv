`timescale 1ns/1ps

module vec_mul_array_engine_tb();

    localparam int N = 4;
    localparam int W = 8;
    localparam int ACC_W = 32;

    logic clk, rst_n;
    logic in_valid, in_ready;
    logic [1:0] op_mode;
    logic [W-1:0] a [N], b [N];
    
    logic out_valid, out_ready;
    logic signed [ACC_W-1:0] out_vec [N];

    // DUT Instantiation
    vec_mul_array_engine #(
        .N(N), .W(W), .ACC_W(ACC_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_ready(in_ready),
        .op_mode(op_mode), .a(a), .b(b),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_vec(out_vec)
    );

    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Task: Feed Data and Check Array
    task run_vec_test(
        input logic [1:0] mode,
        input int val_a [N],
        input int val_b [N],
        input string test_name
    );
        logic signed [ACC_W-1:0] expected_vec [N];
        int i;
        logic pass;

        // 1. Calculate Expected Array
        for (i = 0; i < N; i++) begin
            if (mode == 2'd0)      expected_vec[i] = $signed(val_a[i]) * $signed(val_b[i]);
            else if (mode == 2'd1) expected_vec[i] = $signed(val_a[i]) * $signed({1'b0, val_b[i][W-1:0]});
            else                   expected_vec[i] = $signed({1'b0, val_a[i][W-1:0]}) * $signed({1'b0, val_b[i][W-1:0]});
        end

        // 2. Drive Inputs
        @(posedge clk);
        in_valid <= 1;
        op_mode <= mode;
        for (i = 0; i < N; i++) begin
            a[i] <= val_a[i];
            b[i] <= val_b[i];
        end

        wait(in_ready);
        @(posedge clk);
        in_valid <= 0;

        // 3. Wait for Pipeline
        wait(out_valid);
        @(posedge clk);

        // 4. Array Assertion
        pass = 1'b1;
        for (i = 0; i < N; i++) begin
            if (out_vec[i] !== expected_vec[i]) begin
                $error("[FAIL] %s Lane %0d | Expected: %0d, Got: %0d", test_name, i, expected_vec[i], out_vec[i]);
                pass = 1'b0;
            end
        end
        if (pass) $display("[PASS] %s", test_name);
        else $finish;
    endtask

    // Main Test Sequence
    initial begin
        $display("Starting vec_mul_array_engine Tests...");
        rst_n = 0;
        in_valid = 0;
        out_ready = 1;
        
        #20 rst_n = 1;
        @(posedge clk);

        // Test 1: Signed * Signed
        // Pos*Pos, Pos*Neg, Neg*Pos, Neg*Neg
        run_vec_test(2'd0, '{5, 10, -5, -10}, '{2, -3, 4, -5}, "SS Element-wise Matrix");

        // Test 2: Signed * Unsigned
        // a is signed, b is unsigned (e.g. 255 is positive magnitude)
        run_vec_test(2'd1, '{-10, 20, -30, 40}, '{255, 128, 64, 0}, "SU Element-wise Matrix");

        // Test 3: Unsigned * Unsigned bounds
        // Both unsigned max
        run_vec_test(2'd2, '{255, 200, 100, 0}, '{255, 10, 0, 100}, "UU Element-wise Matrix");

        #50 $display("ALL VEC_MUL TESTS PASSED!");
        $finish;
    end
endmodule