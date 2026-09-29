`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// tr_nonlinear_vpu itself has no single valid_in->valid_out handshake --
// it's a crossbar with FOUR separate valid outputs (bb/mac/vecmul/max),
// driven pass-by-pass externally by the controller (tr_soc_ctrl_int.sv).
// This drives it directly through the SAME 3-pass GELU crossbar sequence
// tr_soc_ctrl_int's GL_P1/GL_P2/GL_P3 states use in production (exact
// mux-select/mode/enable values copied from tr_soc_ctrl_int.sv, re-read
// fresh to get them right), replicating what the controller + its
// external scratch_a/scratch_b registers + the SoC's src_sram_a/b_sel
// muxing do together, entirely within this standalone testbench -- so the
// result is directly comparable to tr_gelu's/ibert_gelu's own per-GELU
// energy numbers.
//
// Counts the TOTAL cycle latency across all 3 passes (Step 1) and captures
// a VCD covering only that clean full-operation window (Steps 2-3). Not
// part of any correctness-verification suite -- this is purely for the
// energy-characterization pilot, and picks GELU as one representative
// operator out of the three tr_nonlinear_vpu can perform (Softmax/RMSNorm
// would need their own separate pilots, not built here).
module tr_nonlinear_vpu_gelu_energy_tb();

    localparam int N = 8, W_VEC = 8, W_MAC = 16, ACC_W = 32, FRAC_W = 4, LUT_IDX_W = 3;

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

    // Pulse en_bb_valid for one cycle, then wait for vpu_bb_valid_out --
    // same "presenting edge not counted, then count until valid" convention
    // used in every other pilot this batch.
    task automatic bb_pass();
        en_bb_valid = 1'b1;
        @(posedge clk);
        en_bb_valid = 1'b0;
        while (!vpu_bb_valid_out) begin @(posedge clk); cycle_count++; end
    endtask

    task automatic vecmul_pass();
        en_vecmul_valid = 1'b1;
        @(posedge clk);
        en_vecmul_valid = 1'b0;
        while (!vpu_vecmul_valid_out) begin @(posedge clk); cycle_count++; end
    endtask

    // Pass 3 gets its own copy of the vecmul wait loop (rather than reusing
    // vecmul_pass()) with a per-cycle settle delay: Pass 3 immediately
    // follows Pass 2's own vecmul_pass() with no bb_pass() in between (the
    // only such back-to-back vecmul transition of the three passes), and
    // reading/driving state in the same zero-delay window as the clock
    // edge that just fired races the simulator's NBA updates for that same
    // edge -- confirmed by hierarchical register tracing during debug
    // (settled reads showed correct values; unsettled reads on the same
    // transition showed Pass 2's stale crossbar output). Pass 1->Pass 2
    // and Pass 2's own bb_pass()+vecmul_pass() aren't back-to-back vecmul
    // calls, so they don't hit this race and don't need the delay.
    task automatic vecmul_pass_settled();
        en_vecmul_valid = 1'b1;
        @(posedge clk); #1;
        en_vecmul_valid = 1'b0;
        while (!vpu_vecmul_valid_out) begin @(posedge clk); #1; cycle_count++; end
    endtask

    initial begin
        logic signed [W_VEC-1:0] x_saved   [N];
        logic signed [W_VEC-1:0] scratch_a [N];  // Pass1 result (E = exp(-alpha|x|))
        logic signed [W_VEC-1:0] scratch_b [N];  // Pass2 result (reciprocal)
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
            x_saved[j]      = W_VEC'(stimulus[j]);
            sram_data_a[j]  = x_saved[j];
        end

        $dumpfile("tr_nonlinear_vpu_gelu_power_activity.vcd");
        $dumpvars(0, tr_nonlinear_vpu_gelu_energy_tb);
        cycle_count = 0;

        // ------ Pass 1 (GL_P1): E = exp(-alpha|x|) ------
        bb_bypass_ln      = 1'b1;
        mux_bb_in_sel     = 3'b001;   // alpha_stab_out
        vecmul_op_mode    = 2'd2;     // UU
        vecmul_scale_mode = 2'b10;    // >>>8
        mux_vecmul_a_sel  = 2'b11;    // rom_e_a
        mux_vecmul_b_sel  = 2'b10;    // mantissa
        mux_vpu_out_sel   = 3'b000;   // vecmul_out_trunc

        bb_pass();
        vecmul_pass();
        for (int j = 0; j < N; j++) scratch_a[j] = vpu_data_out[j];

        // ------ Pass 2 (GL_P2): recip = 1/(1+E) ------
        for (int j = 0; j < N; j++) sram_data_a[j] = scratch_a[j];  // src_sram_a_sel=1 equivalent
        mux_bb_in_sel     = 3'b011;   // raw sram_data_a passthrough
        bb_bypass_ln      = 1'b0;
        bb_mode_pre_ln    = 1'b1;     // +1.0
        bb_mode_post_ln   = 2'b01;    // negate (division)
        vecmul_op_mode    = 2'd2;
        vecmul_scale_mode = 2'b10;
        mux_vecmul_a_sel  = 2'b11;
        mux_vecmul_b_sel  = 2'b10;
        mux_vpu_out_sel   = 3'b000;

        bb_pass();
        vecmul_pass();
        for (int j = 0; j < N; j++) scratch_b[j] = vpu_data_out[j];

        // ------ Pass 3 (GL_P3): y = x * sigma(x) ------
        for (int j = 0; j < N; j++) begin
            sram_data_a[j] = x_saved[j];   // src_sram_a_sel=0 equivalent (original x)
            sram_data_b[j] = scratch_b[j]; // src_sram_b_sel=1 equivalent (reciprocal)
        end
        bb_mode_pre_ln    = 1'b0;
        bb_mode_post_ln   = 2'b00;
        sym_mode_en       = 1'b1;
        vecmul_op_mode    = 2'd1;     // SU
        vecmul_scale_mode = 2'b01;    // >>>4
        mux_vecmul_a_sel  = 2'b00;    // sram_data_a (original x)
        mux_vecmul_b_sel  = 2'b01;    // sym_mod_out (sign-corrected reciprocal)
        mux_vpu_out_sel   = 3'b000;

        vecmul_pass_settled();   // only vecmul this pass, no backbone
        repeat (4) begin @(posedge clk); #1; cycle_count++; end

        $display("[TR_NONLINEAR_VPU GELU ENERGY PILOT] total latency = %0d cycles (across all 3 passes)", cycle_count);
        $display("[TR_NONLINEAR_VPU GELU ENERGY PILOT] vpu_data_out = %p", vpu_data_out);

        $dumpoff;
        $finish;
    end

endmodule
