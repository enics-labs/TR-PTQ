`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives tr_nonlinear_vpu directly through the SAME 4-pass RMSNorm crossbar
// sequence tr_soc_ctrl_int's RM_P1..RM_P4 states use in production (exact
// mux-select/mode/enable values re-read fresh from tr_soc_ctrl_int.sv),
// including the RM_P3 sign-guard (the shared exp backbone only supports
// non-positive inputs, so a positive ln-domain argument is forced negative
// and undone via a small reciprocal LUT when captured into scratch_b --
// mirrored here verbatim from tr_soc_ctrl_int.sv's recip_lut()).
//
// Counts the TOTAL cycle latency across all 4 passes (Step 1) and captures
// a VCD covering only that clean full-operation window (Steps 2-3). Not
// part of any correctness-verification suite -- this is purely for the
// energy-characterization pilot.
module tr_nonlinear_vpu_rmsnorm_energy_tb();

    localparam int N = 8, W_VEC = 8, W_MAC = 16, ACC_W = 32, FRAC_W = 4, LUT_IDX_W = 3;

    localparam logic signed [W_VEC-1:0] CONST_LN_SQRT_N = 8'd17;

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

    // Verbatim from tr_soc_ctrl_int.sv: undoes RM_P3's sign-guard negation
    // when the ln-domain argument was originally positive.
    function automatic logic signed [W_VEC-1:0] recip_lut(input logic signed [W_VEC-1:0] e);
        case (e)
            8'sd0,  8'sd1: recip_lut = 8'sd127;
            8'sd2:         recip_lut = 8'sd127;
            8'sd3:         recip_lut = 8'sd85;
            8'sd4:         recip_lut = 8'sd64;
            8'sd5:         recip_lut = 8'sd51;
            8'sd6:         recip_lut = 8'sd43;
            8'sd7:         recip_lut = 8'sd37;
            8'sd8:         recip_lut = 8'sd32;
            8'sd9:         recip_lut = 8'sd28;
            8'sd10:        recip_lut = 8'sd26;
            8'sd11:        recip_lut = 8'sd23;
            8'sd12:        recip_lut = 8'sd21;
            8'sd13:        recip_lut = 8'sd20;
            8'sd14:        recip_lut = 8'sd18;
            8'sd15:        recip_lut = 8'sd17;
            8'sd16:        recip_lut = 8'sd16;
            8'sd17:        recip_lut = 8'sd15;
            8'sd18:        recip_lut = 8'sd14;
            8'sd19:        recip_lut = 8'sd13;
            8'sd20:        recip_lut = 8'sd13;
            8'sd21:        recip_lut = 8'sd12;
            8'sd22:        recip_lut = 8'sd12;
            8'sd23:        recip_lut = 8'sd11;
            8'sd24:        recip_lut = 8'sd11;
            8'sd25:        recip_lut = 8'sd10;
            8'sd26:        recip_lut = 8'sd10;
            8'sd27:        recip_lut = 8'sd9;
            8'sd28:        recip_lut = 8'sd9;
            8'sd29:        recip_lut = 8'sd9;
            8'sd30:        recip_lut = 8'sd9;
            8'sd31:        recip_lut = 8'sd8;
            default:       recip_lut = 8'sd127;
        endcase
    endfunction

    initial begin
        logic signed [W_VEC-1:0] x_saved   [N];
        logic signed [W_VEC-1:0] scratch_b [N];  // RM_P3 result (InvRMS)
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
        bb_bypass_ln = 0; bb_shift_mode = 0; bb_mode_pre_ln = 0; bb_mode_post_ln = '0; sym_mode_en = 0;
        mux_bb_in_sel = '0; mux_mac_a_sel = '0; mux_mac_b_sel = '0;
        mux_vecmul_a_sel = '0; mux_vecmul_b_sel = '0; mux_vpu_out_sel = '0;
        ctrl_scalar_sub_val = '0;
        for (int i = 0; i < N; i++) begin sram_data_a[i] = '0; sram_data_b[i] = '0; end

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

        // ------ Pass 2 (RM_P2): log_sum = ln(-0.5 * S) = ln(1/sqrt(S)) ------
        reset_ctrl_defaults();
        bb_shift_mode   = 1'b1;   // rms_s_sat (S >> FRAC_W, saturated)
        mux_bb_in_sel   = 3'b000;
        bb_mode_post_ln = 2'b10;  // -0.5x
        mux_vpu_out_sel = 3'b001; // bb_log

        bb_pass();
        // vpu_data_out (MUX6) lags bb_log by one cycle past vpu_bb_valid_out
        // -- see the Softmax pilot's SM_P3 finding.
        @(posedge clk); #1; cycle_count++;
        reg_scalar_log = vpu_data_out[0];

        // ------ Pass 3 (RM_P3): InvRMS = exp(log_sum + ln(sqrt(N))), sign-guarded ------
        reset_ctrl_defaults();
        rm_ctrl_scalar_raw     = reg_scalar_log + CONST_LN_SQRT_N;
        rm_offset_was_positive = ~rm_ctrl_scalar_raw[W_VEC-1] && (rm_ctrl_scalar_raw != '0);
        ctrl_scalar_sub_val     = rm_offset_was_positive ? -rm_ctrl_scalar_raw : rm_ctrl_scalar_raw;
        bb_bypass_ln      = 1'b1;
        mux_bb_in_sel     = 3'b100;   // direct scalar injection (broadcast)
        vecmul_op_mode    = 2'd2;     // UU
        vecmul_scale_mode = 2'b10;    // >>>8
        mux_vecmul_a_sel  = 2'b11;    // rom_e_a
        mux_vecmul_b_sel  = 2'b10;    // mantissa
        mux_vpu_out_sel   = 3'b000;   // vecmul_out_trunc

        bb_pass_settled();   // RM_P2 also ended with bb_pass() -- same-engine back-to-back
        // Resync to a clean clock boundary before the next pulse -- see the
        // Softmax pilot's SM_P4 finding (asserting a new pulse in the same
        // zero-delay window as a #1-delayed statement hangs the sim).
        @(posedge clk); cycle_count++;
        vecmul_pass();
        if (rm_offset_was_positive) begin
            for (int k = 0; k < N; k++) scratch_b[k] = recip_lut(vpu_data_out[k]);
        end else begin
            for (int k = 0; k < N; k++) scratch_b[k] = vpu_data_out[k];
        end

        // ------ Pass 4 (RM_P4): y = x * InvRMS ------
        reset_ctrl_defaults();
        for (int j = 0; j < N; j++) begin
            sram_data_a[j] = x_saved[j];   // src_sram_a_sel=0 equivalent
            sram_data_b[j] = scratch_b[j]; // src_sram_b_sel=1 equivalent
        end
        vecmul_op_mode    = 2'd1;     // SU (signed x * unsigned InvRMS)
        vecmul_scale_mode = 2'b01;    // >>>4, Q4.4 output
        mux_vecmul_a_sel  = 2'b00;    // sram_data_a (x)
        mux_vecmul_b_sel  = 2'b00;    // sram_data_b (InvRMS)
        mux_vpu_out_sel   = 3'b000;

        vecmul_pass_settled();   // RM_P3 also ended with vecmul_pass() -- same-engine back-to-back
        repeat (4) begin @(posedge clk); #1; cycle_count++; end

        $display("[TR_NONLINEAR_VPU RMSNORM ENERGY PILOT] total latency = %0d cycles (across all 4 passes)", cycle_count);
        $display("[TR_NONLINEAR_VPU RMSNORM ENERGY PILOT] reg_scalar_log=%0d rm_offset_was_positive=%0d scratch_b(InvRMS)=%p",
                  reg_scalar_log, rm_offset_was_positive, scratch_b);
        $display("[TR_NONLINEAR_VPU RMSNORM ENERGY PILOT] vpu_data_out = %p", vpu_data_out);

        $dumpoff;
        $finish;
    end

endmodule
