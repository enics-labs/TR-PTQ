`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives tr_nonlinear_vpu directly through the SAME 4-pass Softmax crossbar
// sequence tr_soc_ctrl_int's SM_P1..SM_P4 states use in production (exact
// mux-select/mode/enable values re-read fresh from tr_soc_ctrl_int.sv),
// replicating what the controller + its external reg_scalar_max/
// reg_scalar_log registers do together, entirely within this standalone
// testbench. Unlike GELU/RMSNorm, Softmax's intermediates (row max, log of
// the exp-sum) are SCALARS, not per-lane vectors, so sram_data_a holds the
// original x for the whole sequence -- no scratch-vector re-routing needed.
//
// Counts the TOTAL cycle latency across all 4 passes (Step 1) and captures
// a VCD covering only that clean full-operation window (Steps 2-3). Not
// part of any correctness-verification suite -- this is purely for the
// energy-characterization pilot.
module tr_nonlinear_vpu_softmax_energy_tb();

    localparam int N = 8, W_VEC = 8, W_MAC = 16, ACC_W = 32, FRAC_W = 4, LUT_IDX_W = 3;

    // SM_P4's saturating add of reg_scalar_max + reg_scalar_log, mirrored
    // from tr_soc_ctrl_int.sv's SUB_VAL_MAX/MIN (W=8: +/-2^(W-1)).
    localparam logic signed [W_VEC:0] SUB_VAL_MAX = (1 <<< (W_VEC-1)) - 1;
    localparam logic signed [W_VEC:0] SUB_VAL_MIN = -(1 <<< (W_VEC-1));

    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    logic signed [W_VEC-1:0] sram_data_a [N];
    logic signed [W_VEC-1:0] sram_data_b [N];
    logic signed [W_VEC-1:0] vpu_data_out [N];
    logic signed [W_VEC-1:0] ctrl_scalar_sub_val;
    logic signed [W_VEC-1:0] vpu_max_out;
    logic signed [ACC_W-1:0] vpu_dot_out;

    logic [LUT_IDX_W-1:0] mux_bb_in_sel;
    logic [1:0]           mux_mac_a_sel, mux_mac_b_sel, mux_vecmul_a_sel, mux_vecmul_b_sel;
    logic [LUT_IDX_W-1:0] mux_vpu_out_sel;
    logic                  en_piped_max, en_mac_valid, en_vecmul_valid;
    logic                  mac_clear_acc;
    logic [1:0]            mac_op_mode, vecmul_op_mode, vecmul_scale_mode;
    logic                  en_bb_valid, bb_bypass_ln, bb_shift_mode, bb_mode_pre_ln;
    logic [1:0]            bb_mode_post_ln;
    logic                  sym_mode_en;
    logic                  vpu_bb_valid_out, vpu_vecmul_valid_out, vpu_mac_valid_out, vpu_max_valid_out;

    tr_nonlinear_vpu #(
        .N(N), .W_VEC(W_VEC), .W_MAC(W_MAC), .ACC_W(ACC_W), .FRAC_W(FRAC_W), .LUT_IDX_W(LUT_IDX_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .sram_data_a(sram_data_a), .sram_data_b(sram_data_b), .vpu_data_out(vpu_data_out),
        .ctrl_scalar_sub_val(ctrl_scalar_sub_val),
        .vpu_max_out(vpu_max_out), .vpu_dot_out(vpu_dot_out),
        .mux_bb_in_sel(mux_bb_in_sel), .mux_mac_a_sel(mux_mac_a_sel), .mux_mac_b_sel(mux_mac_b_sel),
        .mux_vecmul_a_sel(mux_vecmul_a_sel), .mux_vecmul_b_sel(mux_vecmul_b_sel), .mux_vpu_out_sel(mux_vpu_out_sel),
        .en_piped_max(en_piped_max), .en_mac_valid(en_mac_valid), .en_vecmul_valid(en_vecmul_valid),
        .mac_clear_acc(mac_clear_acc), .mac_op_mode(mac_op_mode),
        .vecmul_op_mode(vecmul_op_mode), .vecmul_scale_mode(vecmul_scale_mode),
        .en_bb_valid(en_bb_valid), .bb_bypass_ln(bb_bypass_ln), .bb_shift_mode(bb_shift_mode),
        .bb_mode_pre_ln(bb_mode_pre_ln), .bb_mode_post_ln(bb_mode_post_ln),
        .sym_mode_en(sym_mode_en),
        .vpu_bb_valid_out(vpu_bb_valid_out), .vpu_vecmul_valid_out(vpu_vecmul_valid_out),
        .vpu_mac_valid_out(vpu_mac_valid_out), .vpu_max_valid_out(vpu_max_valid_out)
    );

    int unsigned cycle_count;

    // Mirrors tr_soc_ctrl_int.sv's "Safe Global Defaults" block (its
    // always_comb re-zeroes every crossbar control signal each cycle before
    // the current state's case branch applies its own overrides). This
    // testbench's variables are plain procedural state that persists
    // between passes unless explicitly reset, so each pass calls this
    // first to avoid inheriting a previous pass's settings -- e.g. without
    // it, SM_P3 inherited SM_P2's bb_bypass_ln=1 and computed exp() instead
    // of ln(). Note mux_bb_in_sel's real default is 3'b011, not 0.
    task automatic reset_ctrl_defaults();
        mux_bb_in_sel       = 3'b011;
        mux_mac_a_sel       = 2'b00;
        mux_mac_b_sel       = 2'b00;
        mux_vecmul_a_sel    = 2'b00;
        mux_vecmul_b_sel    = 2'b00;
        mux_vpu_out_sel     = 3'b000;
        mac_clear_acc       = 1'b0;
        mac_op_mode         = 2'd0;
        vecmul_op_mode      = 2'd0;
        vecmul_scale_mode   = 2'b00;
        bb_shift_mode       = 1'b0;
        bb_bypass_ln        = 1'b0;
        bb_mode_pre_ln      = 1'b0;
        bb_mode_post_ln     = 2'b00;
        sym_mode_en         = 1'b0;
        ctrl_scalar_sub_val = '0;
    endtask

    task automatic max_pass();
        en_piped_max = 1'b1;
        @(posedge clk);
        en_piped_max = 1'b0;
        while (!vpu_max_valid_out) begin @(posedge clk); cycle_count++; end
    endtask

    task automatic bb_pass();
        en_bb_valid = 1'b1;
        @(posedge clk);
        en_bb_valid = 1'b0;
        while (!vpu_bb_valid_out) begin @(posedge clk); cycle_count++; end
    endtask

    task automatic mac_pass();
        mac_clear_acc = 1'b1;
        en_mac_valid  = 1'b1;
        @(posedge clk);
        mac_clear_acc = 1'b0;
        en_mac_valid  = 1'b0;
        while (!vpu_mac_valid_out) begin @(posedge clk); cycle_count++; end
    endtask

    task automatic vecmul_pass();
        en_vecmul_valid = 1'b1;
        @(posedge clk);
        en_vecmul_valid = 1'b0;
        while (!vpu_vecmul_valid_out) begin @(posedge clk); cycle_count++; end
    endtask

    // SM_P4 immediately follows SM_P3 with no intervening mac/vecmul pass --
    // both are bb_pass()es back-to-back on the SAME backbone engine, the
    // pattern that raced in the GELU pilot's GL_P2->GL_P3 vecmul transition
    // (confirmed there via hierarchical register tracing: a same-engine
    // pass immediately following another one, with no different-engine
    // pass in between to let the free-running pipeline drain, samples the
    // crossbar's stale prior-pass output). Every other pass transition
    // here is preceded by a DIFFERENT engine's pass (max->bb, bb->mac,
    // mac->bb, bb->vecmul), which was already proven safe by the GELU
    // pilot's GL_P1/GL_P2 passes, so only this one needs the settled
    // variant.
    task automatic bb_pass_settled();
        en_bb_valid = 1'b1;
        @(posedge clk); #1;
        en_bb_valid = 1'b0;
        while (!vpu_bb_valid_out) begin @(posedge clk); #1; cycle_count++; end
    endtask

    initial begin
        logic signed [W_VEC-1:0] x_saved [N];
        logic signed [W_VEC-1:0] reg_scalar_max;
        logic signed [W_VEC-1:0] reg_scalar_log;
        logic signed [W_VEC:0]   sub_val_wide;
        int stimulus [N];

        // Same representative vector used for every pilot in this batch.
        stimulus[0]=12; stimulus[1]=-20; stimulus[2]=5; stimulus[3]=-8;
        stimulus[4]=30; stimulus[5]=-3;  stimulus[6]=18; stimulus[7]=-45;

        rst_n = 0;
        en_piped_max = 0; en_mac_valid = 0; en_vecmul_valid = 0; en_bb_valid = 0;
        mac_clear_acc = 0; mac_op_mode = '0; vecmul_op_mode = '0; vecmul_scale_mode = '0;
        bb_bypass_ln = 0; bb_shift_mode = 0; bb_mode_pre_ln = 0; bb_mode_post_ln = '0; sym_mode_en = 0;
        mux_bb_in_sel = '0; mux_mac_a_sel = '0; mux_mac_b_sel = '0;
        mux_vecmul_a_sel = '0; mux_vecmul_b_sel = '0; mux_vpu_out_sel = '0;
        ctrl_scalar_sub_val = '0;
        for (int i = 0; i < N; i++) begin sram_data_a[i] = '0; sram_data_b[i] = '0; end

        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // sram_data_a holds the original x for the entire sequence -- no
        // src_sram_a_sel/src_sram_b_sel re-routing happens in any SM_P* state.
        for (int j = 0; j < N; j++) begin
            x_saved[j]     = W_VEC'(stimulus[j]);
            sram_data_a[j] = x_saved[j];
        end

        $dumpfile("tr_nonlinear_vpu_softmax_power_activity.vcd");
        $dumpvars(0, tr_nonlinear_vpu_softmax_energy_tb);
        cycle_count = 0;

        // ------ Pass 1 (SM_P1): row max ------
        max_pass();
        reg_scalar_max = vpu_max_out;

        // ------ Pass 2 (SM_P2): E = exp(x-max), sum(E) via MAC ------
        reset_ctrl_defaults();
        bb_bypass_ln         = 1'b1;
        ctrl_scalar_sub_val  = reg_scalar_max;
        mux_bb_in_sel        = 3'b010;   // x - max via scalar_sub
        mac_op_mode          = 2'd2;     // UU
        mux_mac_a_sel        = 2'b01;    // e_a
        mux_mac_b_sel        = 2'b01;    // e_frac (mantissa)

        bb_pass();
        mac_pass();

        // ------ Pass 3 (SM_P3): log_sum = ln(sum(E)) ------
        reset_ctrl_defaults();
        bb_shift_mode   = 1'b0;   // use [23:8] slice of vpu_dot_out
        mux_bb_in_sel   = 3'b000; // route vpu_dot_out through the backbone
        mux_vpu_out_sel = 3'b001; // bb_log

        bb_pass();
        // vpu_data_out (MUX6) is a free-running register fed by bb_log,
        // which itself only becomes correct the cycle int_bb_valid first
        // asserts -- reading vpu_data_out immediately after the wait loop
        // exits catches its pre-update (stale) value, same as the GELU
        // pilot's GL_P3 finding. One extra settled cycle fixes it.
        @(posedge clk); #1; cycle_count++;
        reg_scalar_log = vpu_data_out[0];

        // ------ Pass 4 (SM_P4): y = exp(x-(max+log_sum)), Q0.8 unsigned ------
        sub_val_wide = $signed({reg_scalar_max[W_VEC-1], reg_scalar_max}) +
                       $signed({reg_scalar_log[W_VEC-1], reg_scalar_log});
        reset_ctrl_defaults();
        if (sub_val_wide > SUB_VAL_MAX)      ctrl_scalar_sub_val = SUB_VAL_MAX[W_VEC-1:0];
        else if (sub_val_wide < SUB_VAL_MIN) ctrl_scalar_sub_val = SUB_VAL_MIN[W_VEC-1:0];
        else                                  ctrl_scalar_sub_val = sub_val_wide[W_VEC-1:0];
        bb_bypass_ln      = 1'b1;
        mux_bb_in_sel     = 3'b010;
        vecmul_op_mode    = 2'd2;     // UU
        vecmul_scale_mode = 2'b11;    // Q0.8 unsigned, saturating
        mux_vecmul_a_sel  = 2'b11;    // rom_e_a
        mux_vecmul_b_sel  = 2'b10;    // mantissa
        mux_vpu_out_sel   = 3'b000;   // vecmul_out_trunc

        bb_pass_settled();   // SM_P3 also ended with bb_pass() -- same-engine back-to-back
        // Resync to a clean clock boundary before the next pulse: asserting
        // en_vecmul_valid in the same zero-delay window as a #1-delayed
        // statement (bb_pass_settled()'s own trailing #1) hangs the sim.
        @(posedge clk); cycle_count++;
        vecmul_pass();
        repeat (4) begin @(posedge clk); #1; cycle_count++; end

        $display("[TR_NONLINEAR_VPU SOFTMAX ENERGY PILOT] total latency = %0d cycles (across all 4 passes)", cycle_count);
        $display("[TR_NONLINEAR_VPU SOFTMAX ENERGY PILOT] reg_scalar_max=%0d reg_scalar_log=%0d", reg_scalar_max, reg_scalar_log);
        $display("[TR_NONLINEAR_VPU SOFTMAX ENERGY PILOT] vpu_data_out (Q0.8 probabilities) = %p", vpu_data_out);

        $dumpoff;
        $finish;
    end

endmodule
