`timescale 1ns/1ps

// Exhaustive check of shared_lut_rom: every index (0..2**LUT_IDX_W-1) on
// every lane, verifying both the anchor values themselves and the port-
// width plumbing fix (a_idx now genuinely LUT_IDX_W-wide instead of a
// hardcoded [2:0]; e_a now a fixed 8 bits instead of accidentally tied to
// the lane-count parameter N). LUT_IDX_W=3/N=8 matches every one of the 6
// target formats, so no per-format sweep is needed here.
module shared_lut_rom_tb();

    localparam int N         = 8;
    localparam int LUT_IDX_W = 3;

    logic [LUT_IDX_W-1:0] a_idx [N];
    logic [7:0]           e_a   [N];

    shared_lut_rom #(.N(N), .LUT_IDX_W(LUT_IDX_W)) dut (
        .a_idx(a_idx), .e_a(e_a)
    );

    int errors = 0;
    logic [7:0] expected [0:7] = '{94, 35, 13, 5, 2, 1, 0, 0};  // e^-1..e^-8, Q0.8

    initial begin
        $display("=======================================================================");
        $display(" STARTING EXHAUSTIVE SHARED_LUT_ROM VERIFICATION (N=%0d LUT_IDX_W=%0d)", N, LUT_IDX_W);
        $display("=======================================================================");

        for (int idx = 0; idx < (1 << LUT_IDX_W); idx++) begin
            for (int lane = 0; lane < N; lane++) begin
                a_idx[lane] = LUT_IDX_W'(idx);
            end
            #1;
            for (int lane = 0; lane < N; lane++) begin
                logic [7:0] exp_val;
                exp_val = (idx < 8) ? expected[idx] : 8'd0;
                if (e_a[lane] !== exp_val) begin
                    $display("   [FAIL] idx=%0d lane=%0d | Exp: %0d | HW: %0d", idx, lane, exp_val, e_a[lane]);
                    errors++;
                end
            end
        end

        $display("=======================================================================");
        if (errors == 0) begin
            $display(" [SUCCESS] Zero errors across all %0d indices x %0d lanes.", 1 << LUT_IDX_W, N);
        end else begin
            $display(" [FAILED] Found %0d mismatches.", errors);
        end
        $display("=======================================================================");

        $finish;
    end
endmodule
