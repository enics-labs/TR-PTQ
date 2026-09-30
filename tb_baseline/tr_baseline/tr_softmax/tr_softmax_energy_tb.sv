`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives ONE representative vector through the full 6-pass tr_softmax
// sequence (mirrors tr_softmax_tb.sv's own pass structure: max -> sum-exp
// DOT -> ln -> exp-elemwise -> reciprocal -> final-multiply), counts the
// TOTAL cycle latency across all 6 passes (Step 1), and captures a VCD
// covering only that clean full-operation window (Steps 2-3).
//
// Pass 3 (ln(S)) is combinational in the original testbench (no
// valid_out-polling wait, just 2 clock edges to present the operand and
// capture the result) -- kept structurally identical here since that's
// the real cycle cost this specific sequencing takes, not a theoretical
// minimum.
//
// Not part of the correctness-verification suite (see tr_softmax_tb.sv
// for that) -- this is purely for the energy-characterization pilot.
module tr_softmax_energy_tb();
    localparam int N = 8, W = 8, ACC_W = 32;

    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    logic                    valid_in  = 0;
    logic                    valid_out;
    logic [2:0]              mode      = '0;
    logic signed [W-1:0]     x_in      [N];
    logic signed [W-1:0]     offset_in = '0;
    logic signed [ACC_W-1:0] sum_in    = '0;
    logic signed [W-1:0]     y_out     [N];
    logic signed [ACC_W-1:0] sum_out;

    tr_softmax #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) uut (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .valid_out (valid_out),
        .mode      (mode),
        .x_in      (x_in),
        .offset_in (offset_in),
        .sum_in    (sum_in),
        .y_out     (y_out),
        .sum_out   (sum_out)
    );

    int unsigned cycle_count;

    task automatic send_and_wait();
        @(posedge clk); valid_in = 1'b1;
        @(posedge clk); valid_in = 1'b0; cycle_count++;
        while (!valid_out) begin @(posedge clk); cycle_count++; end
    endtask

    initial begin
        int stimulus [N];
        logic signed [W-1:0]     max_val;
        logic signed [ACC_W-1:0] S;
        logic signed [W-1:0]     ln_S;
        logic signed [W-1:0]     exp_vals [N];
        logic signed [W-1:0]     inv_S;

        // Same representative vector used for every pilot in this batch.
        stimulus[0]=12; stimulus[1]=-20; stimulus[2]=5; stimulus[3]=-8;
        stimulus[4]=30; stimulus[5]=-3;  stimulus[6]=18; stimulus[7]=-45;
        for (int j = 0; j < N; j++) x_in[j] = W'(stimulus[j]);

        #20; rst_n = 1;
        @(posedge clk);

        $dumpfile("tr_softmax_power_activity.vcd");
        $dumpvars(0, tr_softmax_energy_tb);
        cycle_count = 0;

        // ------ Pass 1: max(X) ------
        mode = 3'b000; offset_in = '0;
        send_and_wait();
        max_val = y_out[0];

        // ------ Pass 2: exp(xi - max) DOT -> S = sum exp(xi-max) ------
        mode = 3'b001; offset_in = max_val;
        send_and_wait();
        S = sum_out;

        // ------ Pass 3: ln(S) -- combinational, no pipeline wait ------
        mode = 3'b010; sum_in = S;
        @(posedge clk); valid_in = 1'b1; cycle_count++;
        @(posedge clk); ln_S = y_out[0]; valid_in = 1'b0; cycle_count++;

        // ------ Pass 4: exp(xi - max) ELEMWISE -> individual exp values ------
        mode = 3'b011; offset_in = max_val;
        for (int j = 0; j < N; j++) x_in[j] = W'(stimulus[j]);
        send_and_wait();
        for (int j = 0; j < N; j++) exp_vals[j] = y_out[j];

        // ------ Pass 5: exp(0 - ln(S)) = 1/S ------
        mode = 3'b100; offset_in = ln_S;
        for (int j = 0; j < N; j++) x_in[j] = '0;
        send_and_wait();
        inv_S = y_out[0];

        // ------ Pass 6: exp(xi-max) x 1/S -> final probabilities ------
        mode = 3'b101; offset_in = inv_S;
        for (int j = 0; j < N; j++) x_in[j] = exp_vals[j];
        send_and_wait();

        $display("[TR_SOFTMAX ENERGY PILOT] total latency = %0d cycles (across all 6 passes)", cycle_count);
        $display("[TR_SOFTMAX ENERGY PILOT] y_out = %p", y_out);

        @(posedge clk);
        $dumpoff;
        $finish;
    end
endmodule
