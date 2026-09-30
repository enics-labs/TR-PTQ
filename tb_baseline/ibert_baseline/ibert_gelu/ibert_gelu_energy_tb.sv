`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives ONE representative vector through ibert_gelu, counts its exact
// cycle latency (Step 1), and captures a VCD covering only that clean
// operation window -- skipping the reset-adjacent settling cycle and
// stopping right after the result settles -- so a later Genus
// read_activity_file run reflects real switching for this one operation,
// not vectorless guessing or activity averaged in from reset/idle time.
//
// Deliberately NOT part of the correctness-verification suite (see
// ibert_gelu_tb.sv / gen_gelu_golden.py for that, already Xcelium-verified
// 556/556) -- this is purely for the energy-characterization pilot.
module ibert_gelu_energy_tb();

    localparam int N      = 8;
    localparam int W      = 8;
    localparam int FRAC_W = 4;

    logic clk, rst_n;
    logic valid_in, valid_out;
    logic signed [W-1:0] x_in [N];
    logic signed [W-1:0] y_out [N];

    ibert_gelu #(.N(N), .W(W), .FRAC_W(FRAC_W)) dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .x_in(x_in), .valid_out(valid_out), .y_out(y_out)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    int unsigned cycle_count;

    initial begin
        rst_n = 0; valid_in = 0;
        // One representative, non-edge-case, mixed-sign, per-lane-varied
        // vector (Q4.4 codes) -- deliberately not all-zero, not
        // all-saturating, and not identical across lanes, so the
        // lane-parallel switching activity captured below is realistic
        // rather than artificially synchronized (all 8 lanes toggling
        // identically, which an all-broadcast "E"-tag-style vector would
        // give).
        x_in[0] = 8'sd12;  x_in[1] = -8'sd20; x_in[2] = 8'sd5;   x_in[3] = -8'sd8;
        x_in[4] = 8'sd30;  x_in[5] = -8'sd3;  x_in[6] = 8'sd18;  x_in[7] = -8'sd45;

        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Steps 2/3: start the VCD dump only after reset has settled, so
        // reset-transient toggling isn't included in the activity window.
        $dumpfile("power_activity.vcd");
        $dumpvars(0, ibert_gelu_energy_tb);

        // Step 1: cycle-count this one operation's latency, same polling
        // convention as ibert_gelu_tb.sv (while (!valid_out) @(posedge clk)),
        // just with a counter added.
        valid_in = 1'b1;
        @(posedge clk);
        valid_in = 1'b0;
        cycle_count = 0;
        while (!valid_out) begin
            @(posedge clk);
            cycle_count++;
        end

        $display("[GELU ENERGY PILOT] latency = %0d cycles", cycle_count);
        $display("[GELU ENERGY PILOT] y_out = %p", y_out);

        @(posedge clk);   // let the VCD capture the settled-output cycle too
        $dumpoff;
        $finish;
    end

endmodule
