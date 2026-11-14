`timescale 1ns/1ps

// Energy-per-operator pilot of the standalone RMSNorm modules (tr_rmsnorm): one representative 8-element
// tile (the vector used for every pilot in the paper and in tb_soc/tr_nonlinear_vpu/*_energy_tb.sv), the three commands of the flow
// back to back, the exact cycle count from the first command to the last result, and a VCD over exactly that window.
// Energy = Total_Power x N_cycles x T_clock (docs/verification.md). Not part of the correctness suite (see tr_rmsnorm_tb.sv).
module tr_rmsnorm_energy_tb();

    localparam int N = 8, W = 8, OUT_W = 12;

    logic clk = 0, rst_n = 0;
    logic in_valid = 0;
    logic [1:0] cmd = 0;
    logic [3:0] log2n = 3;
    logic signed [W-1:0] x_in [N];
    logic out_valid;
    logic signed [OUT_W-1:0] y_out [N];
    logic [15:0] scale_o;

    tr_rmsnorm #(.N(N), .W(W), .OUT_W(OUT_W)) dut (.*);
    localparam string TAG = "TR_RMSNORM";

    always #5 clk = ~clk;

    // watchdog: a command that never completes ends the run with a message instead of hanging
    initial begin #100000; $display("[TB WATCHDOG] timeout"); $finish; end


    int unsigned cycle_count;
    logic signed [W-1:0] xs [N];

    // issue one command and wait for its completion pulse; every clock edge in between counts toward the latency
    task automatic run(input logic [1:0] c);
        @(posedge clk); #1; cycle_count++;           // inputs change after the edge (no race with the DUT's sampling)
        cmd = c; x_in = xs; in_valid = 1;
        @(posedge clk); #1; cycle_count++;
        in_valid = 0;
        while (!out_valid) begin @(posedge clk); cycle_count++; end
        #1;
    endtask

    initial begin
        xs[0] = 12; xs[1] = -20; xs[2] = 5;  xs[3] = -8;
        xs[4] = 30; xs[5] = -3;  xs[6] = 18; xs[7] = -45;
        for (int i = 0; i < N; i++) x_in[i] = '0;

        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        
        $dumpfile("tr_rmsnorm_power_activity.vcd");
        $dumpvars(0, tr_rmsnorm_energy_tb);
        cycle_count = 0;

        run(2'd0);   // S = sum x^2
        run(2'd2);   // scale register (inv_rms | R)
        run(2'd3);   // y = x * scale
        repeat (2) @(posedge clk);

        $display("[%s ENERGY PILOT] total latency = %0d cycles (ACC_FIRST + SCALE + APPLY of one 8-element tile)", TAG, cycle_count);
        $display("[%s ENERGY PILOT] scale = %0d", TAG, scale_o);
        $display("[%s ENERGY PILOT] y_out = %p", TAG, y_out);
        $dumpoff;
        $finish;
    end

endmodule
