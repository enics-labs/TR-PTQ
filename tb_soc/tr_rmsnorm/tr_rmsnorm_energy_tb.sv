`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives ONE representative vector through the full 4-pass tr_rmsnorm
// sequence (mirrors tr_rmsnorm_tb.sv's own pass structure: sum-of-squares
// -> log/x*gamma -> inverse-RMS reconstruction -> final scaling), counts
// the TOTAL cycle latency across all 4 passes (Step 1), and captures a VCD
// covering only that clean full-operation window (Steps 2-3).
//
// Deliberately omits the original testbench's inter-pass repeat(3)/
// repeat(5) settling delays -- those are display-pacing choices in the
// correctness testbench, not latency the DUT itself requires (its own
// valid_out already gates readiness for the next pass), so including them
// would inflate the cycle count with cycles the real hardware doesn't need.
//
// Not part of the correctness-verification suite (see tr_rmsnorm_tb.sv for
// that) -- this is purely for the energy-characterization pilot.
module tr_rmsnorm_energy_tb();

    localparam int N = 8;
    localparam int W = 8;
    localparam int ACC_W = 32;

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

    int unsigned cycle_count;

    initial begin
        // Same representative vector used for every pilot in this batch;
        // gamma left at 1.0 (Q4.4 code 16) per-lane, same convention as
        // the correctness testbench.
        logic signed [W-1:0] vec   [N];
        logic signed [W-1:0] gamma [N];
        logic signed [ACC_W-1:0] ctrl_sum;
        logic signed [W-1:0]     ctrl_V [N];
        logic signed [W-1:0]     ctrl_log_offset;
        logic signed [W-1:0]     ctrl_inv_rms;

        vec[0]=12; vec[1]=-20; vec[2]=5; vec[3]=-8;
        vec[4]=30; vec[5]=-3;  vec[6]=18; vec[7]=-45;
        for (int i = 0; i < N; i++) gamma[i] = 16;

        rst_n = 0; valid_in = 0; mode = 2'b00; sum_in = '0; offset_in = '0;
        for (int i = 0; i < N; i++) begin x_in[i] = '0; aux_in[i] = '0; end
        #22 rst_n = 1;
        @(posedge clk);

        $dumpfile("tr_rmsnorm_power_activity.vcd");
        $dumpvars(0, tr_rmsnorm_energy_tb);
        cycle_count = 0;

        // ------ Pass 1: sum of squares ------
        @(negedge clk);
        mode = 2'b00; x_in = vec; aux_in = vec; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        do begin @(posedge clk); cycle_count++; end while (!valid_out);
        ctrl_sum = sum_out;

        // ------ Pass 2: log & x*gamma ------
        @(negedge clk);
        mode = 2'b01; x_in = vec; aux_in = gamma; sum_in = ctrl_sum; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        do begin @(posedge clk); cycle_count++; end while (!valid_out);
        for (int i = 0; i < N; i++) ctrl_V[i] = y_out[i];
        ctrl_log_offset = ln_out + CONST_LN_SQRT_N;

        // ------ Pass 3: inverse RMS reconstruction ------
        @(negedge clk);
        mode = 2'b10; offset_in = ctrl_log_offset; valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        do begin @(posedge clk); cycle_count++; end while (!valid_out);
        ctrl_inv_rms = y_out[0];

        // ------ Pass 4: final scaling (V * InvRMS) ------
        @(negedge clk);
        mode = 2'b11; x_in = ctrl_V;
        for (int i = 0; i < N; i++) aux_in[i] = ctrl_inv_rms;
        valid_in = 1'b1;
        @(negedge clk); valid_in = 1'b0;
        do begin @(posedge clk); cycle_count++; end while (!valid_out);

        $display("[TR_RMSNORM ENERGY PILOT] total latency = %0d cycles (across all 4 passes)", cycle_count);
        $display("[TR_RMSNORM ENERGY PILOT] y_out = %p", y_out);

        @(posedge clk);
        $dumpoff;
        $finish;
    end

endmodule
