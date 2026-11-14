`timescale 1ns/1ps

// Energy-per-operator pilot of the standalone softmax tile ibert_softmax: one representative 8-element tile (the vector of the other pilots), the exact
// cycle count from the command to the result, and a VCD over exactly that window. Energy = Total_Power x N_cycles x T_clock
// (docs/verification.md). Not part of the correctness suite (see ibert_softmax_tb.sv).
module ibert_softmax_energy_tb();

    localparam int N = 8, W = 8;

    logic clk = 0, rst_n = 0;
    logic in_valid = 0;
    logic signed [W-1:0] x_in [N];
    logic out_valid;
    logic [W-1:0] p_out [N];

    ibert_softmax #(.N(N), .W(W)) dut (.*);

    always #5 clk = ~clk;

    // watchdog: a tile that never completes ends the run with a message instead of hanging
    initial begin #1000000; $display("[TB WATCHDOG] timeout"); $finish; end

    int unsigned cycle_count;

    initial begin
        x_in[0] = 12; x_in[1] = -20; x_in[2] = 5;  x_in[3] = -8;
        x_in[4] = 30; x_in[5] = -3;  x_in[6] = 18; x_in[7] = -45;

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $dumpfile("ibert_softmax_power_activity.vcd");
        $dumpvars(0, ibert_softmax_energy_tb);
        cycle_count = 0;

        @(posedge clk); #1; cycle_count++;           // inputs change after the edge (no race with the DUT's sampling)
        in_valid = 1;
        @(posedge clk); #1; cycle_count++;
        in_valid = 0;
        while (!out_valid) begin @(posedge clk); cycle_count++; end
        #1;
        repeat (2) @(posedge clk);

        $display("[IBERT_SOFTMAX ENERGY PILOT] total latency = %0d cycles (one 8-element tile, command issue and completion included)", cycle_count);
        $display("[IBERT_SOFTMAX ENERGY PILOT] p_out = %p", p_out);
        $dumpoff;
        $finish;
    end

endmodule
