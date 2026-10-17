`timescale 1ns/1ps

// Pilot testbench for the energy-per-operator methodology (Steps 1-3):
// drives tr_nonlinear_vpu directly through the crossbar sequence of the two-pass SwiGLU of tr_soc_ctrl_int.sv for ONE
// 8-lane group: pass 1 = cmd 5 (silu: x * sigma_rom[x], rounded to the 12-bit silu) + cmd 6 (silu * up through the vec-mul,
// |prod| through the shared piped_max), row gain (cmd 0xB), pass 2 = cmd 5 again + cmd 0xC (apply: round_shift(4*silu*up,
// 10-e) with int8 saturation). Mux-select/mode/enable values are re-read from tr_soc_ctrl_int.sv; the controller-side
// arithmetic (silu rounding, gain function) is mirrored verbatim below.
//
// Counts the TOTAL engine cycle latency of the five passes (Step 1) and captures a VCD covering only that clean
// full-operation window (Steps 2-3). Not part of any correctness-verification suite -- this is purely for the
// energy-characterization pilot. A row of n elements repeats the group sequence n/8 times per pass (plus one gain step).
module tr_nonlinear_vpu_swiglu_energy_tb();

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
    logic                  en_bb_valid, bb_bypass_ln, bb_shift_mode, bb_mode_pre_ln;
    logic [1:0]            bb_mode_post_ln;
    logic                  sym_mode_en;
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
        .bb_mode_pre_ln(bb_mode_pre_ln), .bb_mode_post_ln(bb_mode_post_ln),
        .sym_mode_en(sym_mode_en),
        .vpu_bb_valid_out(vpu_bb_valid_out), .vpu_vecmul_valid_out(vpu_vecmul_valid_out),
        .vpu_mac_valid_out(vpu_mac_valid_out), .vpu_max_valid_out(vpu_max_valid_out)
    );

    int unsigned cycle_count;

    // Mirrors tr_soc_ctrl_int.sv's "Safe Global Defaults" block (see the other pilots for why).
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
        bb_mode_pre_ln      = 1'b0;
        bb_mode_post_ln     = 2'b00;
        sym_mode_en         = 1'b0;
        bb_ln_pre_scaled    = 1'b0;
        mux_max_sel         = 1'b0;
        ctrl_scalar_sub_val = '0;
    endtask

    // Back-to-back vec-mul passes: settled variant (see the GELU pilot's GL_P2->GL_P3 finding).
    task automatic vecmul_pass_settled();
        en_vecmul_valid = 1'b1;
        @(posedge clk); #1;
        en_vecmul_valid = 1'b0;
        while (!vpu_vecmul_valid_out) begin @(posedge clk); #1; cycle_count++; end
    endtask

    task automatic max_pass();
        en_piped_max = 1'b1;
        @(posedge clk); #1;
        en_piped_max = 1'b0;
        while (!vpu_max_valid_out) begin @(posedge clk); #1; cycle_count++; end
    endtask

    // Verbatim from tr_soc_ctrl_int.sv: silu_s = sat12((wide + 2) >> 2).
    function automatic logic signed [11:0] silu_round(input logic signed [19:0] w);
        logic signed [20:0] r;
        r = ($signed({w[19], w}) + 21'sd2) >>> 2;
        if (r > 2047)       silu_round = 12'sd2047;
        else if (r < -2048) silu_round = -12'sd2048;
        else                silu_round = 12'(r);
    endfunction

    // Verbatim from tr_soc_ctrl_int.sv: largest e in [-8,10] with (4*max) * 2^e <= 127*2^10.
    function automatic logic signed [4:0] sw_gain_fn(input logic [19:0] maxp);
        logic [47:0] mx, lim;
        logic signed [4:0] e;
        mx = {maxp, 2'b00};
        if (mx == 0) mx = 1;
        e = -5'sd8;
        for (int k = -8; k <= 10; k++) begin
            lim = 48'd130048;
            if (k >= 0) begin if ((mx << k) <= lim)  e = 5'(k); end
            else        begin if (mx <= (lim << (-k))) e = 5'(k); end
        end
        sw_gain_fn = e;
    endfunction

    // cmd 5 (SW_P1): silu group from the gate codes on sram_data_a.
    task automatic silu_group(input logic signed [W_VEC-1:0] x [N], output logic signed [11:0] silu_s [N]);
        reset_ctrl_defaults();
        for (int j = 0; j < N; j++) sram_data_a[j] = x[j];
        vecmul_op_mode   = 2'd1;      // SU (signed x * unsigned sigma)
        mux_vecmul_a_sel = 2'b00;
        mux_vecmul_b_sel = 3'b100;    // sigma ROM
        vecmul_pass_settled();
        for (int j = 0; j < N; j++) silu_s[j] = silu_round(vpu_wide_out[j]);
        @(posedge clk); #1; cycle_count++;
    endtask

    // silu * up through the vec-mul (cmd 6 / cmd 0xC share this routing); scale mode selects the consumer.
    task automatic product_group(input logic signed [W_VEC-1:0] u [N], input logic signed [11:0] silu_s [N],
                                 input logic [2:0] scale_mode);
        reset_ctrl_defaults();
        for (int j = 0; j < N; j++) begin sram_data_a[j] = u[j]; sw_silu_in[j] = silu_s[j]; end
        vecmul_op_mode    = 2'd0;     // SS
        vecmul_scale_mode = scale_mode;
        mux_vecmul_a_sel  = 2'b00;
        mux_vecmul_b_sel  = 3'b101;   // wide silu
        vecmul_pass_settled();
    endtask

    initial begin
        logic signed [W_VEC-1:0] gate [N], up [N];
        logic signed [11:0]      silu_s [N];
        logic [19:0]             reg_sw_max;
        logic signed [4:0]       reg_sw_e;

        // Representative group: the gate vector shared by the other pilots, plus a varied up vector.
        gate[0]=12; gate[1]=-20; gate[2]=5;  gate[3]=-8; gate[4]=30; gate[5]=-3; gate[6]=18; gate[7]=-45;
        up[0]=40;   up[1]=-25;   up[2]=10;   up[3]=100;  up[4]=-60; up[5]=5;    up[6]=-90;  up[7]=127;

        rst_n = 0;
        en_piped_max = 0; en_mac_valid = 0; en_vecmul_valid = 0; en_bb_valid = 0;
        mac_clear_acc = 0; mac_op_mode = '0; vecmul_op_mode = '0; vecmul_scale_mode = '0;
        bb_bypass_ln = 0; bb_shift_mode = 0; bb_mode_pre_ln = 0; bb_mode_post_ln = '0; sym_mode_en = 0;
        mux_bb_in_sel = '0; mux_mac_a_sel = '0; mux_mac_b_sel = '0;
        mux_vecmul_a_sel = '0; mux_vecmul_b_sel = '0; mux_vpu_out_sel = '0;
        ctrl_scalar_sub_val = '0;
        bb_ln_pre_scaled = 0; mux_max_sel = 0; rms_growth = 0; rms_log2n = 5'd3; inv_rms_in = '0; sw_shift = '0;
        for (int i = 0; i < N; i++) begin sram_data_a[i] = '0; sram_data_b[i] = '0; sw_silu_in[i] = '0; end

        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $dumpfile("tr_nonlinear_vpu_swiglu_power_activity.vcd");
        $dumpvars(0, tr_nonlinear_vpu_swiglu_energy_tb);
        cycle_count = 0;

        // ------ Pass 1, cmd 5 (SW_P1): silu group ------
        silu_group(gate, silu_s);

        // ------ Pass 1, cmd 6 (SW_P2): max |silu*up| through the shared piped_max ------
        product_group(up, silu_s, 3'b000);
        @(posedge clk); #1; cycle_count++;
        reset_ctrl_defaults();
        mux_max_sel = 1'b1;
        max_pass();
        reg_sw_max = vpu_max_out[19:0];

        // ------ Row gain, cmd 0xB (SW_G) ------
        reg_sw_e = sw_gain_fn(reg_sw_max);

        // ------ Pass 2, cmd 5 again, then cmd 0xC (SW_P4): apply ------
        silu_group(gate, silu_s);
        sw_shift = 6'sd10 - 6'(reg_sw_e);
        product_group(up, silu_s, 3'b110);
        repeat (4) begin @(posedge clk); #1; cycle_count++; end

        $display("[TR_NONLINEAR_VPU SWIGLU ENERGY PILOT] total latency = %0d cycles (across the 5 engine passes of one 8-lane group, two-pass)", cycle_count);
        $display("[TR_NONLINEAR_VPU SWIGLU ENERGY PILOT] max|silu*up|=%0d gain e=%0d shift=%0d", reg_sw_max, reg_sw_e, sw_shift);
        $display("[TR_NONLINEAR_VPU SWIGLU ENERGY PILOT] vpu_data_out = %p", vpu_data_out);

        $dumpoff;
        $finish;
    end

endmodule
