`timescale 1ns/1ps

module tb_tr_exp_vector;

    parameter int N = 8;
    parameter int W = 8;
    parameter int ACC_W = 32;

    logic        clk;
    logic        rst_n;
    logic        in_valid;
    logic        in_ready;
    logic [7:0]  x_vector [N];
    logic        clear_acc;
    logic        out_valid;
    logic        out_ready;
    logic [ACC_W-1:0] out_dot;
    

    // --- Clock Generation ---
    initial clk = 0;
    always #5 clk = ~clk;

    // --- DUT Instantiation ---
    tr_exp_high_order_system #(
        .N(N), .W(W), .ACC_W(ACC_W)
    ) dut (.*);

    // --- Stimulus ---
    initial begin
        // Initialize
        rst_n = 0;
        in_valid = 0;
        out_ready = 1;
        clear_acc = 1;
        for(int i=0; i<N; i++) x_vector[i] = 0;

        #20 rst_n = 1;
        #20;

        // Test Case 2: All zeros (e^0 = 1)
        send_vector('{0, 0, 0, 0, 0, 0, 0, 0});

        #100;
        $display("Testbench finished.");
        $finish;
    end

    // --- Data Feeding Task ---
    task send_vector(input logic [7:0] data [N]);
        wait(in_ready);
        @(posedge clk);
        in_valid = 1;
        x_vector = data;
        @(posedge clk);
        in_valid = 0;
    endtask

endmodule