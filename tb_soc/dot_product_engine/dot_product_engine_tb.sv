`timescale 1ns/1ps

module tb_dot_product_engine();

    localparam int M     = 3; // 3 Parallel Lanes
    localparam int N     = 4; // 4 Elements per dot-product
    localparam int W     = 8;
    localparam int ACC_W = 32;

    logic clk, rst_n;
    
    logic in_valid, in_ready;
    logic [1:0] op_mode;
    logic [W-1:0] a_mat [M][N];
    logic [W-1:0] b_vec [N];
    logic signed [ACC_W-1:0] c_vec [M];
    logic clear_acc;

    logic out_valid, out_ready;
    logic signed [ACC_W-1:0] out_vec [M];

    // Instantiation
    dot_product_engine #(
        .M(M), .N(N), .W(W), .ACC_W(ACC_W)
    ) dut (.*);

    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test Sequence
    initial begin
        $display("Starting dot_product_engine Tests...");
        rst_n = 0;
        in_valid = 0;
        out_ready = 1;
        clear_acc = 1;
        op_mode = 2'd0; // Signed x Signed

        #20 rst_n = 1;
        @(posedge clk);

        // ---------------------------------------------------------
        // Test 1: 3x4 Matrix multiplied by 4x1 Vector + Bias
        // ---------------------------------------------------------
        
        // Setup B Vector (Broadcasted Activations)
        b_vec = '{10, 10, 10, 10};

        // Lane 0: A = {1, 1, 1, 1}, Bias = 0   => Expected: 40
        a_mat[0] = '{1, 1, 1, 1};
        c_vec[0] = 0;

        // Lane 1: A = {2, -2, 2, -2}, Bias = 100 => Expected: (20 - 20 + 20 - 20) + 100 = 100
        a_mat[1] = '{2, -2, 2, -2};
        c_vec[1] = 100;

        // Lane 2: A = {-5, -5, -5, -5}, Bias = 50 => Expected: -200 + 50 = -150
        a_mat[2] = '{-5, -5, -5, -5};
        c_vec[2] = 50;

        // Fire transaction
        in_valid <= 1;
        wait(in_ready);
        @(posedge clk);
        in_valid <= 0;

        // Wait for pipeline
        wait(out_valid);
        @(posedge clk);

        // Assertions
        if (out_vec[0] === 32'd40 && out_vec[1] === 32'd100 && out_vec[2] === -32'sd150) begin
            $display("[PASS] Matrix-Vector Multiply successful across all M lanes!");
        end else begin
            $error("[FAIL] Matrix-Vector Multiply failed. Got: %0d, %0d, %0d", 
                   out_vec[0], out_vec[1], out_vec[2]);
        end

        #20 $finish;
    end

endmodule