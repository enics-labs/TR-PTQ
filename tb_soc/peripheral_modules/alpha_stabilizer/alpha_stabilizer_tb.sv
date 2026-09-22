`timescale 1ns/1ps

module tb_alpha_stabilizer();

    localparam int N      = 8;
    localparam int W      = 8;
    localparam int FRAC_W = 4;

    // DUT Signals
    logic signed [W-1:0] in_vec [N];
    logic signed [W-1:0] out_vec [N];

    // Instantiate the DUT
    alpha_stabilizer #(.N(N), .W(W), .FRAC_W(FRAC_W)) dut (
        .in_vec  (in_vec),
        .out_vec (out_vec)
    );

    int errors = 0;

    initial begin
        $display("=======================================================================");
        $display(" STARTING EXHAUSTIVE ALPHA STABILIZER VERIFICATION (W=%0d FRAC_W=%0d, ALL %0d STATES)", W, FRAC_W, 1 << W);
        $display("=======================================================================");

        // We only need to test Lane 0 since all lanes are structurally identical
        for (int i = 1; i < N; i++) begin
            in_vec[i] = '0;
        end

        // Sweep every possible W-bit signed value
        for (int test_val = -(1 <<< (W-1)); test_val <= (1 <<< (W-1)) - 1; test_val++) begin

            logic signed [W-1:0] hw_in;
            logic signed [W-1:0] hw_out;
            logic signed [W-1:0] exp_out;

            hw_in = W'(test_val);

            // 1. Drive the hardware
            in_vec[0] = hw_in;
            #1; // Wait for combinational logic to settle

            hw_out = out_vec[0];

            // 2. Calculate the Golden Expected Value
            exp_out = golden_model(hw_in);

            // 3. Compare and Report
            if (hw_out !== exp_out) begin
                $display("   \033[0;31m[FAIL]\033[0m Input: %4d | Exp: %4d | HW: %4d", hw_in, exp_out, hw_out);
                errors++;
            end
        end

        $display("=======================================================================");
        if (errors == 0) begin
            $display(" \033[0;32m[SUCCESS]\033[0m Zero errors across all %0d states (W=%0d FRAC_W=%0d).", 1 << W, W, FRAC_W);
        end else begin
            $display(" \033[0;31m[FAILED]\033[0m Found %0d mismatches.", errors);
        end
        $display("=======================================================================");

        $finish;
    end

    // ========================================================================
    // BEHAVIORAL GOLDEN MODEL -- plain int/longint arithmetic, independent of
    // the DUT's own bit-slice/shift-add tricks. Region boundaries are the
    // real |x| = 1, 2, 3 thresholds at this format's LSB (reduces to the
    // original's abs_x[6:4] bit-slice exactly at W=8, FRAC_W=4).
    // ========================================================================
    function automatic logic signed [W-1:0] golden_model(input logic signed [W-1:0] x);
        longint abs_x, multiplier, raw_prod, scaled_val, final_val;
        longint max_out, min_out;

        abs_x = (x < 0) ? -longint'(x) : longint'(x);
        max_out = (1 <<< (W-1)) - 1;
        min_out = -(1 <<< (W-1));

        if (abs_x < (1 <<< FRAC_W))      multiplier = 27 <<< (FRAC_W - 4);
        else if (abs_x < (2 <<< FRAC_W)) multiplier = 26 <<< (FRAC_W - 4);
        else if (abs_x < (3 <<< FRAC_W)) multiplier = 25 <<< (FRAC_W - 4);
        else                             multiplier = 24 <<< (FRAC_W - 4);

        raw_prod = longint'(x) * multiplier;

        if (raw_prod > (max_out <<< FRAC_W)) begin
            scaled_val = max_out;
        end else if (raw_prod < (min_out <<< FRAC_W)) begin
            scaled_val = min_out;
        end else begin
            // Arithmetic right shift by FRAC_W (mimics hardware slice [11:4] at W=8,FRAC_W=4)
            scaled_val = raw_prod >>> FRAC_W;
        end

        // Force Negative Absolute Value
        final_val = (scaled_val > 0) ? -scaled_val : scaled_val;

        return W'(final_val);
    endfunction

endmodule
