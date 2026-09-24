`timescale 1ns/1ps

module tr_nonlinear_vpu_tb();

    // ---------------------------------------------------------
    // Parameters -- W_VEC/FRAC_W/RM_CONST_LN_SQRT_N edited here per format
    // before each run, same convention as the synthesis top.
    // RM_CONST_LN_SQRT_N = round(0.5*ln(8) * 2^FRAC_W), the RMSNorm
    // controller constant this testbench injects directly (that constant
    // lives in tr_soc_ctrl_int.sv, out of scope for this module -- see
    // sweep_tr_nonlinear_vpu.sh / gen_vpu_golden.py for the derivation).
    // ---------------------------------------------------------
    localparam int N                 = 8;
    localparam int W_VEC             = 8;
    localparam int W_MAC             = 16;
    localparam int ACC_W             = 32;
    localparam int FRAC_W            = 4;
    localparam int LUT_IDX_W         = 3;
    localparam int RM_CONST_LN_SQRT_N = 17;

    // ---------------------------------------------------------
    // DUT Signals
    // ---------------------------------------------------------
    logic clk, rst_n;

    // SRAM Interfaces
    logic signed [W_VEC-1:0] sram_data_a [N];
    logic signed [W_VEC-1:0] sram_data_b [N];
    logic signed [W_VEC-1:0] vpu_data_out [N];

    // Controller Registers
    logic signed [W_VEC-1:0] ctrl_scalar_sub_val;
    logic signed [W_VEC-1:0] vpu_max_out;
    logic signed [ACC_W-1:0] vpu_dot_out;

    // MUX Controls
    logic [LUT_IDX_W-1:0] mux_bb_in_sel;
    logic [1:0] mux_mac_a_sel;
    logic [1:0] mux_mac_b_sel;
    logic [1:0] mux_vecmul_a_sel;
    logic [1:0] mux_vecmul_b_sel;
    logic [LUT_IDX_W-1:0] mux_vpu_out_sel;

    // Enables & Modes
    logic       en_piped_max;
    logic       en_mac_valid;
    logic       en_vecmul_valid;
    logic       en_bb_valid;

    logic       vpu_bb_valid_out;
    logic       vpu_mac_valid_out;
    logic       vpu_vecmul_valid_out;
    logic       vpu_max_valid_out;

    logic       mac_clear_acc;
    logic [1:0] mac_op_mode;
    logic [1:0] vecmul_op_mode;
    logic [1:0] vecmul_scale_mode;

    logic       bb_shift_mode;
    logic       bb_bypass_ln;
    logic       bb_mode_pre_ln;
    logic [1:0] bb_mode_post_ln;
    logic       sym_mode_en;

    // ---------------------------------------------------------
    // DUT Instantiation
    // ---------------------------------------------------------
    tr_nonlinear_vpu #(
        .N(N),
        .W_VEC(W_VEC),
        .W_MAC(W_MAC),
        .ACC_W(ACC_W),
        .FRAC_W(FRAC_W),
        .LUT_IDX_W(LUT_IDX_W)
    ) dut (.*); // SystemVerilog wildcard connection

    // ---------------------------------------------------------
    // Clock Generation
    // ---------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ---------------------------------------------------------
    // Helper Task: Reset all control lines to safe zero states
    // ---------------------------------------------------------
    task automatic clear_crossbar();
        mux_bb_in_sel    = '0;
        mux_mac_a_sel    = '0;
        mux_mac_b_sel    = '0;
        mux_vecmul_a_sel = '0;
        mux_vecmul_b_sel = '0;
        mux_vpu_out_sel  = '0;

        en_piped_max    = 0;
        en_mac_valid    = 0;
        en_vecmul_valid = 0;
        en_bb_valid     = 0;
        mac_clear_acc   = 0;
        mac_op_mode     = 0;
        vecmul_op_mode  = 0;
        vecmul_scale_mode = 0;

        bb_shift_mode   = 0;
        bb_bypass_ln    = 0;
        bb_mode_pre_ln  = 0;
        bb_mode_post_ln = 0;
        sym_mode_en     = 0;

        ctrl_scalar_sub_val = '0;
        for (int i=0; i<N; i++) begin
            sram_data_a[i] = '0;
            sram_data_b[i] = '0;
        end
    endtask

    // ---------------------------------------------------------
    // Pulse-an-enable-and-wait-for-its-valid helpers -- mirrors
    // tr_soc_ctrl_int.sv's own "en=1 for one cycle, then poll valid"
    // pattern, robust to whatever pipeline depth each engine has.
    // Each one first waits for its own valid to be LOW: the real FSM
    // naturally spends >=1 cycle in a fresh state (enable held low by
    // default) between any two passes, but back-to-back calls to the SAME
    // helper across passes (e.g. GL_P2's and GL_P3's do_vecmul(), with no
    // intervening do_bb() on a different signal) have no such gap here --
    // without this wait, the second call's poll loop can see the FIRST
    // call's still-asserted valid and return immediately with the stale
    // result instead of the new one.
    // ---------------------------------------------------------
    task automatic do_max();
        while (vpu_max_valid_out) @(posedge clk);
        en_piped_max = 1'b1;
        @(posedge clk);
        en_piped_max = 1'b0;
        while (!vpu_max_valid_out) @(posedge clk);
    endtask

    task automatic do_bb();
        while (vpu_bb_valid_out) @(posedge clk);
        en_bb_valid = 1'b1;
        @(posedge clk);
        en_bb_valid = 1'b0;
        while (!vpu_bb_valid_out) @(posedge clk);
    endtask

    task automatic do_mac();
        while (vpu_mac_valid_out) @(posedge clk);
        en_mac_valid = 1'b1;
        @(posedge clk);
        en_mac_valid = 1'b0;
        while (!vpu_mac_valid_out) @(posedge clk);
    endtask

    task automatic do_vecmul();
        while (vpu_vecmul_valid_out) @(posedge clk);
        en_vecmul_valid = 1'b1;
        @(posedge clk);
        en_vecmul_valid = 1'b0;
        while (!vpu_vecmul_valid_out) @(posedge clk);
    endtask

    // ---------------------------------------------------------
    // GELU: GL_P1 -> GL_P2 -> GL_P3 (tr_soc_ctrl_int.sv's own sequence,
    // replicated directly -- tr_nonlinear_vpu has no state of its own, so
    // the inter-pass scratch_a/scratch_b capture the controller would do
    // is done here in the testbench instead).
    // ---------------------------------------------------------
    task automatic run_gelu(input logic signed [W_VEC-1:0] x_in [N], output logic signed [W_VEC-1:0] y_out [N]);
        logic signed [W_VEC-1:0] scratch_a [N];
        logic signed [W_VEC-1:0] scratch_b [N];

        // GL_P1
        clear_crossbar();
        sram_data_a = x_in;
        bb_bypass_ln = 1'b1;
        mux_bb_in_sel = 3'b001;
        vecmul_op_mode = 2'd2;
        vecmul_scale_mode = 2'b10;
        mux_vecmul_a_sel = 2'b11;
        mux_vecmul_b_sel = 2'b10;
        mux_vpu_out_sel = 3'b000;
        do_bb();
        do_vecmul();
        scratch_a = vpu_data_out;

        // GL_P2 (src_sram_a_sel=1 -> reads scratch_a)
        clear_crossbar();
        sram_data_a = scratch_a;
        mux_bb_in_sel = 3'b011;
        bb_mode_pre_ln = 1'b1;
        bb_mode_post_ln = 2'b01;
        vecmul_op_mode = 2'd2;
        vecmul_scale_mode = 2'b10;
        mux_vecmul_a_sel = 2'b11;
        mux_vecmul_b_sel = 2'b10;
        mux_vpu_out_sel = 3'b000;
        do_bb();
        do_vecmul();
        scratch_b = vpu_data_out;

        // GL_P3 (src_sram_a_sel=0 -> raw x; src_sram_b_sel=1 -> scratch_b)
        clear_crossbar();
        sram_data_a = x_in;
        sram_data_b = scratch_b;
        sym_mode_en = 1'b1;
        vecmul_op_mode = 2'd1;
        vecmul_scale_mode = 2'b01;
        mux_vecmul_a_sel = 2'b00;
        mux_vecmul_b_sel = 2'b01;
        mux_vpu_out_sel = 3'b000;
        do_vecmul();
        y_out = vpu_data_out;
    endtask

    // ---------------------------------------------------------
    // Softmax: SM_P1 -> SM_P2 -> SM_P3 -> SM_P4. reg_scalar_max/log and the
    // saturating add are controller registers -- replicated here (the
    // saturating-add bounds are already W-parameterized in the real FSM,
    // so this is a faithful port, not a simplification).
    // ---------------------------------------------------------
    task automatic run_softmax(input logic signed [W_VEC-1:0] x_in [N], output logic [W_VEC-1:0] y_out [N], output logic signed [W_VEC-1:0] z_out);
        logic signed [W_VEC-1:0] reg_scalar_max;
        logic signed [W_VEC-1:0] reg_scalar_log;
        logic signed [W_VEC:0]   sub_val_wide;
        logic signed [W_VEC:0]   sub_val_max, sub_val_min;

        sub_val_max = (1 <<< (W_VEC-1)) - 1;
        sub_val_min = -(1 <<< (W_VEC-1));

        // SM_P1: max
        clear_crossbar();
        sram_data_a = x_in;
        do_max();
        reg_scalar_max = vpu_max_out;

        // SM_P2: sum_i exp(x_i - max) via bb(bypass_ln) -> MAC(UU)
        clear_crossbar();
        sram_data_a = x_in;
        bb_bypass_ln = 1'b1;
        ctrl_scalar_sub_val = reg_scalar_max;
        mux_bb_in_sel = 3'b010;
        mac_op_mode = 2'd2;
        mux_mac_a_sel = 2'b01;
        mux_mac_b_sel = 2'b01;
        do_bb();
        mac_clear_acc = 1'b1;
        do_mac();

        // SM_P3: ln(sum) -- ingest from vpu_dot_out (mux_bb_in_sel=000, bb_shift_mode=0 -> >>>8)
        clear_crossbar();
        bb_shift_mode = 1'b0;
        mux_bb_in_sel = 3'b000;
        mux_vpu_out_sel = 3'b001;
        do_bb();
        @(posedge clk);  // vpu_data_out (registered from bb_log every cycle) lags vpu_bb_valid_out by 1 cycle when read with no intervening vecmul/mac stage
        reg_scalar_log = vpu_data_out[0];

        // SM_P4: sub_val2 = sat(max+log), exp(x_i - sub_val2) -> Q0.8 unsigned
        clear_crossbar();
        sram_data_a = x_in;
        bb_bypass_ln = 1'b1;
        sub_val_wide = $signed({reg_scalar_max[W_VEC-1], reg_scalar_max}) + $signed({reg_scalar_log[W_VEC-1], reg_scalar_log});
        if (sub_val_wide > sub_val_max) ctrl_scalar_sub_val = sub_val_max[W_VEC-1:0];
        else if (sub_val_wide < sub_val_min) ctrl_scalar_sub_val = sub_val_min[W_VEC-1:0];
        else ctrl_scalar_sub_val = sub_val_wide[W_VEC-1:0];
        z_out = ctrl_scalar_sub_val;
        mux_bb_in_sel = 3'b010;
        vecmul_op_mode = 2'd2;
        vecmul_scale_mode = 2'b11;
        mux_vecmul_a_sel = 2'b11;
        mux_vecmul_b_sel = 2'b10;
        mux_vpu_out_sel = 3'b000;
        do_bb();
        do_vecmul();
        for (int i = 0; i < N; i++) y_out[i] = vpu_data_out[i];
    endtask

    // ---------------------------------------------------------
    // RMSNorm, both branches: RM_P1 -> RM_P2 -> RM_P3 -> RM_P4, mirroring
    // tr_soc_ctrl_int.sv's FSM (incl. its sign-guard: a positive
    // ctrl_scalar is negated before the backbone and undone by a
    // reciprocal on the way into scratch_b).
    //
    // The reciprocal itself is CONTROLLER logic (tr_soc_ctrl_int.sv's
    // recip_lut(), 8-bit / Q4.4-only), not part of tr_nonlinear_vpu, and
    // the controller is not parameterized. recip_ideal() below emulates
    // it: round(2^(2*FRAC_W)/max(E,1)) saturated to [0, 2^(W_VEC-1)-1]. At
    // Q4.4 it equals recip_lut() for every E in 0..31 (checked in
    // gen_vpu_golden_from_hw.py); at the other formats it is the
    // specification the model uses, with no controller RTL behind it yet.
    // ---------------------------------------------------------
    function automatic logic signed [W_VEC-1:0] recip_ideal(input logic signed [W_VEC-1:0] e);
        longint unsigned num, den, r, hi;
        den = (e < 1) ? 64'd1 : longint'(e);
        num = 64'd1 << (2*FRAC_W);
        r   = (num + den/2) / den;
        hi  = (64'd1 << (W_VEC-1)) - 1;
        recip_ideal = (r > hi) ? W_VEC'(hi) : W_VEC'(r);
    endfunction

    task automatic run_rmsnorm(input logic signed [W_VEC-1:0] x_in [N], output logic signed [W_VEC-1:0] y_out [N]);
        logic signed [W_VEC-1:0] reg_scalar_log;
        logic signed [W_VEC-1:0] scratch_b [N];
        logic signed [W_VEC-1:0] rm_ctrl_scalar_raw;
        logic                    was_positive;

        // RM_P1: sum of squares
        clear_crossbar();
        sram_data_a = x_in;
        sram_data_b = x_in;
        mux_mac_a_sel = 2'b00;
        mux_mac_b_sel = 2'b10;
        mac_op_mode = 2'd0;
        mac_clear_acc = 1'b1;
        do_mac();

        // RM_P2: -0.5*ln(sum) -- ingest from vpu_dot_out, bb_shift_mode=1 -> >>>FRAC_W
        clear_crossbar();
        bb_shift_mode = 1'b1;
        mux_bb_in_sel = 3'b000;
        bb_mode_post_ln = 2'b10;
        mux_vpu_out_sel = 3'b001;
        do_bb();
        @(posedge clk);  // vpu_data_out lags vpu_bb_valid_out by 1 cycle with no intervening vecmul/mac stage
        reg_scalar_log = vpu_data_out[0];

        // RM_P3: reconstruct InvRMS
        rm_ctrl_scalar_raw = reg_scalar_log + RM_CONST_LN_SQRT_N[W_VEC-1:0];
        was_positive = ~rm_ctrl_scalar_raw[W_VEC-1] && (rm_ctrl_scalar_raw != '0);

        clear_crossbar();
        ctrl_scalar_sub_val = was_positive ? -rm_ctrl_scalar_raw : rm_ctrl_scalar_raw;
        // src_sram_a_sel=1, reading empty scratch (never written) == 0
        for (int i = 0; i < N; i++) sram_data_a[i] = '0;
        bb_bypass_ln = 1'b1;
        mux_bb_in_sel = 3'b100;
        vecmul_op_mode = 2'd2;
        vecmul_scale_mode = 2'b10;
        mux_vecmul_a_sel = 2'b11;
        mux_vecmul_b_sel = 2'b10;
        mux_vpu_out_sel = 3'b000;
        do_bb();
        do_vecmul();
        for (int i = 0; i < N; i++) scratch_b[i] = was_positive ? recip_ideal(vpu_data_out[i]) : vpu_data_out[i];

        // RM_P4: X * InvRMS
        clear_crossbar();
        sram_data_b = scratch_b;
        vecmul_op_mode = 2'd1;
        vecmul_scale_mode = 2'b01;
        mux_vecmul_a_sel = 2'b00;
        mux_vecmul_b_sel = 2'b00;
        mux_vpu_out_sel = 3'b000;
        for (int i = 0; i < N; i++) sram_data_a[i] = x_in[i];
        do_vecmul();
        y_out = vpu_data_out;
    endtask

    // ---------------------------------------------------------
    // File-I/O driven multi-format regression. Format per line:
    // "op x0 x1 ... x7" where op is 0=GELU, 1=SOFTMAX, 2=RMSNORM_DECAY, 3=RMSNORM_GROWTH.
    // Writes "y0 y1 ... y7 [z]" to hdl_out.txt (z only for softmax, else 0).
    // ---------------------------------------------------------
    task automatic run_file_regression();
        int file_in, file_out, num_vecs, dummy;
        int op;
        int stim [N];
        logic signed [W_VEC-1:0] x_in [N];
        logic signed [W_VEC-1:0] y_out_s [N];
        logic [W_VEC-1:0]        y_out_u [N];
        logic signed [W_VEC-1:0] z_out;

        file_in = $fopen("inputs.txt", "r");
        file_out = $fopen("hdl_out.txt", "w");
        if (!file_in || !file_out) begin
            $display("[ERROR] Could not open IO files!");
            $finish;
        end

        dummy = $fscanf(file_in, "%d\n", num_vecs);
        for (int v = 0; v < num_vecs; v++) begin
            dummy = $fscanf(file_in, "%d %d %d %d %d %d %d %d %d\n", op, stim[0], stim[1], stim[2], stim[3], stim[4], stim[5], stim[6], stim[7]);
            for (int i = 0; i < N; i++) x_in[i] = W_VEC'(stim[i]);

            z_out = '0;
            if (op == 0) begin
                run_gelu(x_in, y_out_s);
                $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d %0d\n", $signed(y_out_s[0]), $signed(y_out_s[1]), $signed(y_out_s[2]), $signed(y_out_s[3]), $signed(y_out_s[4]), $signed(y_out_s[5]), $signed(y_out_s[6]), $signed(y_out_s[7]), 0);
            end else if (op == 1) begin
                run_softmax(x_in, y_out_u, z_out);
                $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d %0d\n", y_out_u[0], y_out_u[1], y_out_u[2], y_out_u[3], y_out_u[4], y_out_u[5], y_out_u[6], y_out_u[7], $signed(z_out));
            end else begin
                run_rmsnorm(x_in, y_out_s);
                $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d %0d\n", $signed(y_out_s[0]), $signed(y_out_s[1]), $signed(y_out_s[2]), $signed(y_out_s[3]), $signed(y_out_s[4]), $signed(y_out_s[5]), $signed(y_out_s[6]), $signed(y_out_s[7]), 0);
            end
        end

        $fclose(file_in);
        $fclose(file_out);
        $display("[RTL SIM] Successfully evaluated %0d vectors at W_VEC=%0d FRAC_W=%0d.", num_vecs, W_VEC, FRAC_W);
    endtask

    // ---------------------------------------------------------
    // Main Test Sequence
    // ---------------------------------------------------------
    initial begin
        $display("==================================================");
        $display("  Starting VPU Crossbar Routing Tests...          ");
        $display("==================================================");

        rst_n = 0;
        clear_crossbar();
        #20 rst_n = 1;
        @(posedge clk);

        // TESTS 1-4 use hardcoded Q4.4 expected values -- only valid at the
        // default format; skip them for the other 5 sweep formats instead
        // of spuriously failing.
        if (W_VEC == 8 && FRAC_W == 4) begin

        // ====================================================================
        // TEST 1: Pure VecMul Bypass (SRAM -> VecMul -> SRAM)
        // Emulates: Generic scaling or parallel RMSNorm Affine phase
        // ====================================================================
        $display("\n[TEST 1] Routing: SRAM -> VecMul -> SRAM");
        clear_crossbar();

        mux_vecmul_a_sel = 2'b00; // SRAM A
        mux_vecmul_b_sel = 1'b0;  // SRAM B
        mux_vpu_out_sel  = 3'b000; // VecMul Out
        vecmul_op_mode   = 2'd0;  // Signed x Signed

        @(posedge clk);
        en_vecmul_valid = 1;
        sram_data_a = '{2, -3, 4, -5, 0, 0, 0, 0};
        sram_data_b = '{10, 10, 10, 10, 0, 0, 0, 0};

        @(posedge clk); en_vecmul_valid = 0;
        repeat(4) @(posedge clk);

        if (vpu_data_out[0] === 8'd20 && vpu_data_out[1] === -8'sd30)
            $display("  -> [PASS] VecMul routed correctly to SRAM output.");
        else
            $error("  -> [FAIL] VecMul routing failed.");

        // ====================================================================
        // TEST 2: Dense MAC Reduction (SRAM -> MAC -> Controller Dot)
        // ====================================================================
        $display("\n[TEST 2] Routing: SRAM -> MAC Engine -> Controller Dot Reg");
        clear_crossbar();

        mux_mac_a_sel = 2'b00;
        mux_mac_b_sel = 2'b00;
        mac_clear_acc = 1'b1;
        mac_op_mode   = 2'd0;

        @(posedge clk);
        en_mac_valid = 1;
        sram_data_a = '{2, 3, 4, 5, 0, 0, 0, 0};
        sram_data_b = '{2, 3, 4, 5, 0, 0, 0, 0}; // 4+9+16+25 = 54

        @(posedge clk); en_mac_valid = 0;
        repeat(5) @(posedge clk);

        if (vpu_dot_out === 32'd54)
            $display("  -> [PASS] MAC Sum of Squares routed to Controller Reg.");
        else
            $error("  -> [FAIL] MAC reduction routing failed. Got: %0d", vpu_dot_out);

        // ====================================================================
        // TEST 3: Log-Domain Scalar Injection (Dot Reg -> TR-Backbone -> SRAM)
        // ====================================================================
        $display("\n[TEST 3] Routing: Dot Reg -> TR-Backbone -> SRAM");
        clear_crossbar();

        mux_bb_in_sel   = 3'b000;
        bb_mode_post_ln = 2'b01;
        mux_vpu_out_sel = 3'b001;

        @(posedge clk);
        en_bb_valid = 1;

        @(posedge clk);
        en_bb_valid = 0;

        do begin @(posedge clk); end while (!vpu_bb_valid_out);

        if (vpu_data_out[0] !== 8'hxx)
            $display("  -> [PASS] Pipelined backbone routing is active and propagating.");
        else
            $error("  -> [FAIL] Pipelined backbone routing is disconnected or stuck.");

        // ====================================================================
        // TEST 4: GELU Symmetry Gating (SRAM -> Sym Mod -> VecMul -> SRAM)
        // ====================================================================
        $display("\n[TEST 4] Routing: SRAM -> Symmetry Modifier -> VecMul -> SRAM");
        clear_crossbar();

        mux_vecmul_a_sel = 2'b00;
        mux_vecmul_b_sel = 2'b01;
        sym_mode_en      = 1'b1;
        mux_vpu_out_sel  = 3'b000;
        vecmul_op_mode   = 2'd0;

        @(posedge clk);
        en_vecmul_valid = 1;
        sram_data_a = '{-10, 10, 0, 0, 0, 0, 0, 0};
        sram_data_b = '{5, 6, 0, 0, 0, 0, 0, 0};

        @(posedge clk); en_vecmul_valid = 0;
        repeat(4) @(posedge clk);

        if (vpu_data_out[0] === -8'sd110 && vpu_data_out[1] === 8'd60)
            $display("  -> [PASS] Complex GELU Symmetry routing successful.");
        else
            $error("  -> [FAIL] GELU routing failed. L0: %0d, L1: %0d", vpu_data_out[0], vpu_data_out[1]);

        end // W_VEC==8 && FRAC_W==4

        // ====================================================================
        // TEST 5: Full-sequence multi-format regression (file I/O)
        // ====================================================================
        $display("\n[TEST 5] Full GELU/Softmax/RMSNorm(decay+growth) sequences vs golden model");
        clear_crossbar();
        @(posedge clk);
        run_file_regression();

        $display("\n==================================================");
        $display("  ALL VPU CROSSBAR TESTS COMPLETED.               ");
        $display("==================================================");
        $finish;
    end

endmodule
