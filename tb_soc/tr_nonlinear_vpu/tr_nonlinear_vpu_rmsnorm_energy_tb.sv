`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives tr_nonlinear_vpu directly through the crossbar sequence the tiled RMSNorm of tr_soc_ctrl_int.sv uses
// for ONE 8-element tile (RMS_LOG2N = 3): RM_P1 (sum of squares, cmd 7), RM_P2 (mean -> shared ln at BITS=FRAC_W+2
// -> rounded -0.5x, cmd 9), RM_P3 (sign-guarded exp -> wide inv_rms, cmd 9; the growth-branch reciprocal is
// selected inside the VPU by rms_growth), RM_P4 (rounded x * inv_rms, cmd 0xA). Mux-select/mode/enable values
// are re-read from tr_soc_ctrl_int.sv.
//
// Counts the TOTAL cycle latency across all 4 passes (Step 1) and captures
// a VCD covering only that clean full-operation window (Steps 2-3). Not
// part of any correctness-verification suite -- this is purely for the
// energy-characterization pilot.
module tr_nonlinear_vpu_rmsnorm_energy_tb();

    localparam int N = 8, W_VEC = 8, W_MAC = 16, ACC_W = 32, FRAC_W = 4, LUT_IDX_W = 3;

    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    logic signed [W_VEC-1:0] sram_data_a [N];
    logic signed [W_VEC-1:0] sram_data_b [N];
    logic signed [W_VEC-1:0] vpu_data_out [N];
    logic signed [W_VEC-1:0] ctrl_scalar_sub_val;
    logic signed [19:0]      vpu_max_out;   // WIDE_W: shared piped_max (softmax row max / SwiGLU max |prod|)
    logic signed [ACC_W-1:0] vpu_dot_out;

    logic [LUT_IDX_W-1:0] mux_bb_in_sel;
    logic [1:0]           mux_mac_a_sel, mux_mac_b_sel, mux_vecmul_a_sel;
    logic [2:0]           mux_vecmul_b_sel;
    logic [LUT_IDX_W-1:0] mux_vpu_out_sel;
    logic                  en_piped_max, en_mac_valid, en_vecmul_valid;
    logic                  mac_clear_acc;
    logic [1:0]            mac_op_mode, vecmul_op_mode;
    logic [2:0]            vecmul_scale_mode;
    logic                  en_bb_valid, bb_bypass_ln, bb_shift_mode;
    logic [1:0]            bb_mode_post_ln;
    // Newer VPU ports (shared max select, RMSNorm inv_rms, SwiGLU); idle in this pilot unless the sequence uses them.
    logic                  bb_ln_pre_scaled, mux_max_sel, rms_growth;
    logic [4:0]            rms_log2n;
    logic [11:0]           inv_rms_in, vpu_inv_rms;
    logic signed [11:0]    sw_silu_in [N];
    logic signed [5:0]     sw_shift;
    logic signed [19:0]    vpu_wide_out [N];
    logic                  vpu_bb_valid_out, vpu_vecmul_valid_out, vpu_mac_valid_out, vpu_max_valid_out;

    tr_nonlinear_vpu #(
        .N(N), .W_VEC(W_VEC), .W_MAC(W_MAC), .ACC_W(ACC_W), .FRAC_W(FRAC_W), .LUT_IDX_W(LUT_IDX_W)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .sram_data_a(sram_data_a), .sram_data_b(sram_data_b), .vpu_data_out(vpu_data_out),
        .ctrl_scalar_sub_val(ctrl_scalar_sub_val),
        .vpu_max_out(vpu_max_out), .vpu_dot_out(vpu_dot_out),
        .rms_growth(rms_growth), .vpu_inv_rms(vpu_inv_rms), .rms_log2n(rms_log2n), .inv_rms_in(inv_rms_in),
        .sw_silu_in(sw_silu_in), .sw_shift(sw_shift), .vpu_wide_out(vpu_wide_out),
        .mux_bb_in_sel(mux_bb_in_sel), .mux_mac_a_sel(mux_mac_a_sel), .mux_mac_b_sel(mux_mac_b_sel),
        .mux_vecmul_a_sel(mux_vecmul_a_sel), .mux_vecmul_b_sel(mux_vecmul_b_sel), .mux_vpu_out_sel(mux_vpu_out_sel),
        .en_piped_max(en_piped_max), .mux_max_sel(mux_max_sel), .bb_ln_pre_scaled(bb_ln_pre_scaled), .en_mac_valid(en_mac_valid), .en_vecmul_valid(en_vecmul_valid),
        .mac_clear_acc(mac_clear_acc), .mac_op_mode(mac_op_mode),
        .vecmul_op_mode(vecmul_op_mode), .vecmul_scale_mode(vecmul_scale_mode),
        .en_bb_valid(en_bb_valid), .bb_bypass_ln(bb_bypass_ln), .bb_shift_mode(bb_shift_mode),
        .bb_mode_post_ln(bb_mode_post_ln),
        .vpu_bb_valid_out(vpu_bb_valid_out), .vpu_vecmul_valid_out(vpu_vecmul_valid_out),
        .vpu_mac_valid_out(vpu_mac_valid_out), .vpu_max_valid_out(vpu_max_valid_out)
    );

    int unsigned cycle_count;

    // Mirrors tr_soc_ctrl_int.sv's "Safe Global Defaults" block -- see the
    // Softmax pilot for why this is needed (this testbench's variables
    // persist between passes unless explicitly reset, unlike the real
    // controller's always_comb which re-zeroes everything every cycle).
    // Note mux_bb_in_sel's real default is 3'b011, not 0.
    task automatic reset_ctrl_defaults();
        mux_bb_in_sel       = 3'b011;
        mux_mac_a_sel       = 2'b00;
        mux_mac_b_sel       = 2'b00;
        mux_vecmul_a_sel    = 2'b00;
        mux_vecmul_b_sel    = 3'b000;
        mux_vpu_out_sel     = 3'b000;
        mac_clear_acc       = 1'b0;
        mac_op_mode         = 2'd0;
        vecmul_op_mode      = 2'd0;
        vecmul_scale_mode   = 3'b000;
        bb_shift_mode       = 1'b0;
        bb_bypass_ln        = 1'b0;
        bb_mode_post_ln     = 2'b00;
        bb_ln_pre_scaled    = 1'b0;
        mux_max_sel         = 1'b0;
        ctrl_scalar_sub_val = '0;
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

    // RM_P2->RM_P3 is bb->bb back-to-back (no intervening mac/vecmul pass),
    // the same same-engine hazard found in the GELU (GL_P2->GL_P3, vecmul)
    // and Softmax (SM_P3->SM_P4, bb) pilots -- see those for the root
    // cause. Used for RM_P3's own bb sub-pass.
    task automatic bb_pass_settled();
        en_bb_valid = 1'b1;
        @(posedge clk); #1;
        en_bb_valid = 1'b0;
        while (!vpu_bb_valid_out) begin @(posedge clk); #1; cycle_count++; end
    endtask

    // RM_P3->RM_P4 is vecmul->vecmul back-to-back -- the exact GL_P2->GL_P3
    // pattern from the GELU pilot. Used for RM_P4's own vecmul pass.
    task automatic vecmul_pass_settled();
        en_vecmul_valid = 1'b1;
        @(posedge clk); #1;
        en_vecmul_valid = 1'b0;
        while (!vpu_vecmul_valid_out) begin @(posedge clk); #1; cycle_count++; end
    endtask

    // RM_P1 is the very first engine pass in this sequence (no bb/max pass
    // precedes it), yet the plain mac_pass() still raced: vpu_dot_out was
    // still 0 the exact cycle vpu_mac_valid_out first asserted (confirmed
    // by hierarchical tracing during debug), one cycle before out_dot
    // itself settles. RM_P2 reads vpu_dot_out combinationally right after,
    // so RM_P1 needs the settled variant too.
    task automatic mac_pass_settled();
        mac_clear_acc = 1'b1;
        en_mac_valid  = 1'b1;
        @(posedge clk); #1;
        mac_clear_acc = 1'b0;
        en_mac_valid  = 1'b0;
        while (!vpu_mac_valid_out) begin @(posedge clk); #1; cycle_count++; end
    endtask

    initial begin
        logic signed [W_VEC-1:0] x_saved   [N];
        logic [11:0]             inv_rms;        // RM_P3 result (wide InvRMS, from vpu_inv_rms)
        logic signed [W_VEC-1:0] reg_scalar_log;
        logic signed [W_VEC-1:0] rm_ctrl_scalar_raw;
        logic                    rm_offset_was_positive;
        int stimulus [N];

        // Same representative vector used for every pilot in this batch.
        stimulus[0]=12; stimulus[1]=-20; stimulus[2]=5; stimulus[3]=-8;
        stimulus[4]=30; stimulus[5]=-3;  stimulus[6]=18; stimulus[7]=-45;

        rst_n = 0;
        en_piped_max = 0; en_mac_valid = 0; en_vecmul_valid = 0; en_bb_valid = 0;
        mac_clear_acc = 0; mac_op_mode = '0; vecmul_op_mode = '0; vecmul_scale_mode = '0;
        bb_bypass_ln = 0; bb_shift_mode = 0; bb_mode_post_ln = '0;
        mux_bb_in_sel = '0; mux_mac_a_sel = '0; mux_mac_b_sel = '0;
        mux_vecmul_a_sel = '0; mux_vecmul_b_sel = '0; mux_vpu_out_sel = '0;
        ctrl_scalar_sub_val = '0;
        bb_ln_pre_scaled = 0; mux_max_sel = 0; rms_growth = 0; rms_log2n = 5'd3; inv_rms_in = '0; sw_shift = '0;
        for (int i = 0; i < N; i++) begin sram_data_a[i] = '0; sram_data_b[i] = '0; sw_silu_in[i] = '0; end

        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        for (int j = 0; j < N; j++) begin
            x_saved[j]     = W_VEC'(stimulus[j]);
            sram_data_a[j] = x_saved[j];
        end

        $dumpfile("tr_nonlinear_vpu_rmsnorm_power_activity.vcd");
        $dumpvars(0, tr_nonlinear_vpu_rmsnorm_energy_tb);
        cycle_count = 0;

        // ------ Pass 1 (RM_P1): S = sum(x*x) ------
        reset_ctrl_defaults();
        mux_mac_a_sel = 2'b00;  // x
        mux_mac_b_sel = 2'b10;  // x (case 2'b10 for mac_in_b routes sram_data_a)
        mac_op_mode   = 2'd0;   // SS

        mac_pass_settled();
        @(posedge clk); cycle_count++;  // resync before RM_P2's bb_pass() pulse (see mac_pass_settled comment)

        // ------ Pass 2 (RM_P2): mk = round(S >> (FRAC_W+log2N-2)) -> ln -> rounded -0.5x ------
        reset_ctrl_defaults();
        bb_shift_mode    = 1'b1;   // mean of squares (rms_s_sat)
        bb_ln_pre_scaled = 1'b1;   // mk already has FRAC_W+2 fractional bits
        mux_bb_in_sel    = 3'b000;
        bb_mode_post_ln  = 2'b11;  // rounded -0.5x
        mux_vpu_out_sel  = 3'b001; // bb_log

        bb_pass();
        // vpu_data_out (MUX6) lags bb_log by one cycle past vpu_bb_valid_out
        // -- see the Softmax pilot's SM_P3 finding.
        @(posedge clk); #1; cycle_count++;
        reg_scalar_log = vpu_data_out[0];

        // ------ Pass 3 (RM_P3): InvRMS = exp(reg_scalar_log), sign-guarded ------
        reset_ctrl_defaults();
        rm_ctrl_scalar_raw     = reg_scalar_log;
        rm_offset_was_positive = ~rm_ctrl_scalar_raw[W_VEC-1] && (rm_ctrl_scalar_raw != '0);
        rms_growth             = rm_offset_was_positive;   // selects the reciprocal path inside the VPU
        ctrl_scalar_sub_val    = rm_offset_was_positive ? -rm_ctrl_scalar_raw : rm_ctrl_scalar_raw;
        bb_bypass_ln      = 1'b1;
        mux_bb_in_sel     = 3'b100;   // direct scalar injection (broadcast)
        vecmul_op_mode    = 2'd2;     // UU
        vecmul_scale_mode = 3'b010;   // raw product is consumed through vpu_inv_rms
        mux_vecmul_a_sel  = 2'b11;    // rom_e_a
        mux_vecmul_b_sel  = 3'b010;   // mantissa
        mux_vpu_out_sel   = 3'b000;   // vecmul_out_trunc

        bb_pass_settled();   // RM_P2 also ended with bb_pass() -- same-engine back-to-back
        // Resync to a clean clock boundary before the next pulse -- see the
        // Softmax pilot's SM_P4 finding (asserting a new pulse in the same
        // zero-delay window as a #1-delayed statement hangs the sim).
        @(posedge clk); cycle_count++;
        vecmul_pass();
        inv_rms = vpu_inv_rms;       // the controller latches this into reg_inv_rms (latch_inv)

        // ------ Pass 4 (RM_P4): y = round(x * InvRMS >> (FRAC_W+4)), int8 saturate ------
        reset_ctrl_defaults();
        rms_growth = 1'b0;
        inv_rms_in = inv_rms;                                  // reg_inv_rms broadcast
        for (int j = 0; j < N; j++) sram_data_a[j] = x_saved[j];   // src_sram_a_sel=0 equivalent
        vecmul_op_mode    = 2'd1;     // SU (signed x * unsigned wide InvRMS)
        vecmul_scale_mode = 3'b101;   // rounded >>(FRAC_W+4), signed saturate
        mux_vecmul_a_sel  = 2'b00;    // sram_data_a (x)
        mux_vecmul_b_sel  = 3'b011;   // inv_rms_in
        mux_vpu_out_sel   = 3'b000;

        vecmul_pass_settled();   // RM_P3 also ended with vecmul_pass() -- same-engine back-to-back
        repeat (4) begin @(posedge clk); #1; cycle_count++; end

        $display("[TR_NONLINEAR_VPU RMSNORM ENERGY PILOT] total latency = %0d cycles (across all 4 passes of one 8-element tile)", cycle_count);
        $display("[TR_NONLINEAR_VPU RMSNORM ENERGY PILOT] reg_scalar_log=%0d rm_offset_was_positive=%0d inv_rms=%0d",
                  reg_scalar_log, rm_offset_was_positive, inv_rms);
        $display("[TR_NONLINEAR_VPU RMSNORM ENERGY PILOT] vpu_data_out = %p", vpu_data_out);

        $dumpoff;
        $finish;
    end

endmodule
