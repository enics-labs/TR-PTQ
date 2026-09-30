`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives ONE representative vector through the full 3-pass tr_gelu
// sequence (mirrors tr_gelu_tb.sv's own pass structure: exp(-alpha|x|) ->
// reciprocal -> gate multiply, since tr_gelu is pass-based, not a single
// valid_in->valid_out hop), counts the TOTAL cycle latency across all 3
// passes (Step 1), and captures a VCD covering only that clean
// full-operation window (Steps 2-3).
//
// Not part of the correctness-verification suite (see tr_gelu_tb.sv for
// that) -- this is purely for the energy-characterization pilot.
module tr_gelu_energy_tb();

    localparam int N    = 8;
    localparam int W    = 8;
    localparam int ACC_W = 32;

    logic                clk = 0;
    logic                rst_n = 0;
    logic                valid_in = 0;
    logic [1:0]          mode = '0;
    logic signed [W-1:0] x_in   [N];
    logic signed [W-1:0] aux_in [N];
    logic                valid_out;
    logic signed [W-1:0] y_out  [N];

    tr_gelu #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) dut (.*);

    always #5 clk = ~clk;

    int unsigned cycle_count;

    task automatic send_and_wait();
        @(posedge clk); valid_in = 1'b1;
        @(posedge clk); valid_in = 1'b0; cycle_count++;
        while (!valid_out) begin @(posedge clk); cycle_count++; end
    endtask

    initial begin
        logic signed [W-1:0] x_saved    [N];
        logic signed [W-1:0] ctrl_E     [N];
        logic signed [W-1:0] ctrl_recip [N];
        int stimulus [N];

        // Same representative, non-edge-case, mixed-sign vector used for
        // every pilot in this batch (Q4.4 codes).
        stimulus[0]=12; stimulus[1]=-20; stimulus[2]=5; stimulus[3]=-8;
        stimulus[4]=30; stimulus[5]=-3;  stimulus[6]=18; stimulus[7]=-45;

        for (int i = 0; i < N; i++) begin x_in[i] = '0; aux_in[i] = '0; end
        #20; rst_n = 1;
        @(posedge clk);

        for (int j = 0; j < N; j++) begin
            x_saved[j] = W'(stimulus[j]);
            x_in[j]    = x_saved[j];
            aux_in[j]  = '0;
        end

        // Steps 2/3: dump only the clean operation window.
        $dumpfile("tr_gelu_power_activity.vcd");
        $dumpvars(0, tr_gelu_energy_tb);
        cycle_count = 0;

        // ------ Pass 0: E = exp(-alpha|x|) ------
        mode = 2'b00;
        send_and_wait();
        for (int j = 0; j < N; j++) ctrl_E[j] = y_out[j];

        // ------ Pass 1: recip = 1/(1+E) ------
        mode = 2'b01;
        for (int j = 0; j < N; j++) begin x_in[j] = ctrl_E[j]; aux_in[j] = '0; end
        send_and_wait();
        for (int j = 0; j < N; j++) ctrl_recip[j] = y_out[j];

        // ------ Pass 2: y = x * sigma(x) ------
        mode = 2'b10;
        for (int j = 0; j < N; j++) begin x_in[j] = x_saved[j]; aux_in[j] = ctrl_recip[j]; end
        send_and_wait();

        $display("[TR_GELU ENERGY PILOT] total latency = %0d cycles (across all 3 passes)", cycle_count);
        $display("[TR_GELU ENERGY PILOT] y_out = %p", y_out);

        @(posedge clk);
        $dumpoff;
        $finish;
    end

endmodule
