`timescale 1ns/1ps

// Energy-per-operator pilot of the standalone SwiGLU module tr_swiglu: one representative 8-lane group (the gate vector of the other pilots and
// the up vector of tb_soc/tr_nonlinear_vpu/tr_nonlinear_vpu_swiglu_energy_tb.sv), the five commands of the two-pass flow back to back
// (SILU, MAX, GAIN, SILU, APPLY), the exact cycle count from the first command to the last result, and a VCD over exactly that window.
// Energy = Total_Power x N_cycles x T_clock (docs/verification.md). Not part of the correctness suite (see tr_swiglu_tb.sv).
module tr_swiglu_energy_tb();

    localparam int N = 8, W = 8, OUT_W = 12;

    logic clk = 0, rst_n = 0;
    logic in_valid = 0;
    logic [1:0] cmd = 0;
    logic signed [W-1:0] x_in [N];
    logic out_valid;
    logic signed [OUT_W-1:0] y_out [N];
    logic [15:0] scale_o;

    tr_swiglu #(.N(N), .W(W), .OUT_W(OUT_W)) dut (.*);

    always #5 clk = ~clk;

    // watchdog: a command that never completes ends the run with a message instead of hanging
    initial begin #1000000; $display("[TB WATCHDOG] timeout"); $finish; end

    int unsigned cycle_count;
    logic signed [W-1:0] gate [N], up [N];

    // issue one command and wait for its completion pulse; every clock edge in between counts toward the latency
    task automatic run(input logic [1:0] c, input logic signed [W-1:0] x [N]);
        @(posedge clk); #1; cycle_count++;           // inputs change after the edge (no race with the DUT's sampling)
        cmd = c; x_in = x; in_valid = 1;
        @(posedge clk); #1; cycle_count++;
        in_valid = 0;
        while (!out_valid) begin @(posedge clk); cycle_count++; end
        #1;
    endtask

    initial begin
        gate[0] = 12; gate[1] = -20; gate[2] = 5;  gate[3] = -8; gate[4] = 30; gate[5] = -3; gate[6] = 18; gate[7] = -45;
        up[0]   = 40; up[1]   = -25; up[2]   = 10; up[3]   = 100; up[4]  = -60; up[5]  = 5;  up[6]  = -90; up[7]  = 127;
        for (int i = 0; i < N; i++) x_in[i] = '0;

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $dumpfile("tr_swiglu_power_activity.vcd");
        $dumpvars(0, tr_swiglu_energy_tb);
        cycle_count = 0;

        run(2'd0, gate);   // pass 1: silu of the gate group
        run(2'd1, up);     //         running max of |silu * up|
        run(2'd2, up);     // row gain
        run(2'd0, gate);   // pass 2: silu again
        run(2'd3, up);     //         apply
        repeat (2) @(posedge clk);

        $display("[TR_SWIGLU ENERGY PILOT] total latency = %0d cycles (SILU + MAX + GAIN + SILU + APPLY of one 8-lane group)", cycle_count);
        $display("[TR_SWIGLU ENERGY PILOT] gain e = %0d", $signed(scale_o));
        $display("[TR_SWIGLU ENERGY PILOT] y_out = %p", y_out);
        $dumpoff;
        $finish;
    end

endmodule
