`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives ONE representative vector through ibert_softmax, counts its exact
// cycle latency (Step 1), and captures a VCD covering only that clean
// operation window (Steps 2-3). Expect this latency to be notably larger
// than ibert_gelu's -- ibert_softmax runs up to 8 SEQUENTIAL divisions
// through the shared iterative (non-pipelined) ibert_divider, one per lane.
//
// Not part of the correctness-verification suite (see ibert_softmax_tb.sv
// for that) -- this is purely for the energy-characterization pilot.
module ibert_softmax_energy_tb();

    localparam int N      = 8;
    localparam int W      = 8;
    localparam int FRAC_W = 4;

    logic clk, rst_n;
    logic valid_in, valid_out;
    logic signed [W-1:0] x_in [N];
    logic [W-1:0]        y_out [N];

    ibert_softmax #(.N(N), .W(W), .FRAC_W(FRAC_W)) dut (
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
        // Same representative, non-edge-case, mixed-sign vector used for
        // every pilot in this batch (Q4.4 codes).
        x_in[0] = 8'sd12;  x_in[1] = -8'sd20; x_in[2] = 8'sd5;   x_in[3] = -8'sd8;
        x_in[4] = 8'sd30;  x_in[5] = -8'sd3;  x_in[6] = 8'sd18;  x_in[7] = -8'sd45;

        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $dumpfile("ibert_softmax_power_activity.vcd");
        $dumpvars(0, ibert_softmax_energy_tb);

        valid_in = 1'b1;
        @(posedge clk);
        valid_in = 1'b0;
        cycle_count = 0;
        while (!valid_out) begin
            @(posedge clk);
            cycle_count++;
        end

        $display("[IBERT_SOFTMAX ENERGY PILOT] latency = %0d cycles", cycle_count);
        $display("[IBERT_SOFTMAX ENERGY PILOT] y_out = %p", y_out);

        @(posedge clk);
        $dumpoff;
        $finish;
    end

endmodule
