`timescale 1ns/1ps

module tb_alpha_stabilizer();

    localparam int N      = 8;
    localparam int W      = 8;
    localparam int FRAC_W = 4;

    // DUT Signals
    logic signed [W-1:0] in_vec [N];
    logic                flat_mode;
    logic signed [W-1:0] out_vec [N];

    // Instantiate the DUT
    alpha_stabilizer #(.N(N), .W(W), .FRAC_W(FRAC_W)) dut (
        .in_vec    (in_vec),
        .flat_mode (flat_mode),
        .out_vec   (out_vec)
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

        // Pass 1: flat_mode=0 -- must be BIT-IDENTICAL to the pre-existing
        // piecewise GELU-tuned behavior (proves the new port is zero-impact
        // on every existing caller, GELU included, when unused).
        flat_mode = 1'b0;
        for (int test_val = -(1 <<< (W-1)); test_val <= (1 <<< (W-1)) - 1; test_val++) begin
            logic signed [W-1:0] hw_in;
            logic signed [W-1:0] hw_out;
            logic signed [W-1:0] exp_out;

            hw_in = W'(test_val);
            in_vec[0] = hw_in;
            #1;
            hw_out = out_vec[0];
            exp_out = golden_model(hw_in, 1'b0);

            if (hw_out !== exp_out) begin
                $display("   \033[0;31m[FAIL flat_mode=0]\033[0m Input: %4d | Exp: %4d | HW: %4d", hw_in, exp_out, hw_out);
                errors++;
            end
        end
        $display(" flat_mode=0 sweep done (%0d errors so far).", errors);

        // Pass 2: flat_mode=1 -- real alpha=1 sigmoid slope (SwiGLU's SiLU,
        // see docs/llama_fpga_plan.md change A). Exhaustive over all W-bit
        // values.
        flat_mode = 1'b1;
        for (int test_val = -(1 <<< (W-1)); test_val <= (1 <<< (W-1)) - 1; test_val++) begin
            logic signed [W-1:0] hw_in;
            logic signed [W-1:0] hw_out;
            logic signed [W-1:0] exp_out;

            hw_in = W'(test_val);
            in_vec[0] = hw_in;
            #1;
            hw_out = out_vec[0];
            exp_out = golden_model(hw_in, 1'b1);

            if (hw_out !== exp_out) begin
                $display("   \033[0;31m[FAIL flat_mode=1]\033[0m Input: %4d | Exp: %4d | HW: %4d", hw_in, exp_out, hw_out);
                errors++;
            end
        end
        $display(" flat_mode=1 sweep done (%0d errors total).", errors);

        $display("=======================================================================");
        if (errors == 0) begin
            $display(" \033[0;32m[SUCCESS]\033[0m Zero errors across both modes, all %0d states each (W=%0d FRAC_W=%0d).", 1 << W, W, FRAC_W);
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
    function automatic logic signed [W-1:0] golden_model(input logic signed [W-1:0] x, input logic flat_mode);
        longint abs_x, multiplier, raw_prod, scaled_val, final_val;
        longint max_out, min_out;

        abs_x = (x < 0) ? -longint'(x) : longint'(x);
        max_out = (1 <<< (W-1)) - 1;
        min_out = -(1 <<< (W-1));

        if (flat_mode)                   multiplier = 1 <<< FRAC_W;
        else if (abs_x < (1 <<< FRAC_W))      multiplier = 27 <<< (FRAC_W - 4);
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
