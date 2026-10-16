/*
 * @module   tr_nonlinear_vpu
 * @brief    Software-Defined Vector Processing Unit.
 * @details  A strictly combinational/pipelined crossbar that routes SRAM data
 *           through isolated math modifiers, a log-domain backbone, and local
 *           vector/MAC engines. Contains ZERO internal state memory.
 *
 * @param    N          Vector dimension (lanes).
 * @param    W_VEC      SRAM/vector data width (sram_data_a/b, vpu_data_out,
 *                        modifier and math-engine operand width).
 * @param    W_MAC      Internal log/exp backbone working width (wider than
 *                        W_VEC, since ln/exp intermediates need headroom).
 * @param    ACC_W      MAC/vecmul accumulator width (vpu_dot_out, vecmul's
 *                        internal accumulation before rescale to W_VEC).
 * @param    FRAC_W     Fractional bits of the Q(W_VEC-FRAC_W).FRAC_W format.
 * @param    LUT_IDX_W  Width of the shared exp-LUT index, reused here as the
 *                        crossbar select width for mux_bb_in_sel/mux_vpu_out_sel.
 * @param    INV_W      Width of RMSNorm's wide inv_rms (unsigned, FRAC_W+INV_BITS fractional bits);
 *                        feeds the vecmul b operand through mux_vecmul_b_sel==2'b11.
 * @param    RMS_K      Extra fractional bits kept on RMSNorm's mean-of-squares (= tr_backbone_wrapper LN_EXTRA).
 */
module tr_nonlinear_vpu #(
    parameter int N         = 8,
    parameter int W_VEC     = 8,
    parameter int W_MAC     = 16,
    parameter int ACC_W     = 32,
    parameter int FRAC_W    = 4,
    parameter int LUT_IDX_W = 3,
    parameter int INV_W     = 12,
    parameter int RMS_K     = 2
)(
    input  logic clk,
    input  logic rst_n,

    // =========================================================
    // External Memory & Controller Interfaces
    // =========================================================
    input  logic signed [W_VEC-1:0] sram_data_a [N],
    input  logic signed [W_VEC-1:0] sram_data_b [N],
    output logic signed [W_VEC-1:0] vpu_data_out [N],
    input  logic signed [W_VEC-1:0] ctrl_scalar_sub_val, // For x - max or x - mu
    output logic signed [W_VEC-1:0] vpu_max_out,         // To Controller Reg
    output logic signed [ACC_W-1:0] vpu_dot_out,         // To Controller Reg
    output logic        [15:0]      vpu_exp_raw,         // vecmul lane-0 raw product (e_a*mantissa), for RMSNorm inv_rms
    input  logic        [4:0]       rms_log2n,           // RMSNorm: log2(row length), sets the mean shift
    input  logic        [INV_W-1:0] inv_rms_in,          // RMSNorm: wide inv_rms (broadcast vecmul b operand)

    // =========================================================
    // MX Expansion Ports (Conditionally Compiled)
    // =========================================================
`ifdef MX_MODE
    input  logic signed [7:0]       mx_shared_exp,
    input  logic mx_shift_en,
`endif

    // =========================================================
    // Crossbar Routing Controls
    // =========================================================
    input  logic [LUT_IDX_W-1:0]  mux_bb_in_sel,   // Backbone Input MUX
    input  logic [1:0]  mux_mac_a_sel,   // MAC Input A MUX
    input  logic [1:0]  mux_mac_b_sel,   // MAC Input B MUX
    input  logic [1:0]  mux_vecmul_a_sel,// VecMul Input A MUX
    input  logic [1:0]  mux_vecmul_b_sel,// VecMul Input B MUX
    input  logic [LUT_IDX_W-1:0]  mux_vpu_out_sel, // Final Output MUX to SRAM

    // =========================================================
    // Submodule Enables & Modes
    // =========================================================
    input  logic en_piped_max,
    input  logic en_mac_valid,
    input  logic en_vecmul_valid,
    input  logic mac_clear_acc,
    input  logic [1:0]  mac_op_mode,
    input  logic [1:0]  vecmul_op_mode,
    // 000:[N-1:0], 001:[11:4] (wrap), 010:[15:8], 011: softmax Q0.8 unsigned
    // saturate, 101: ROUNDED >>(FRAC_W+4) signed saturate (RMSNorm apply),
    // 100: [11:4] SIGNED SATURATE (SwiGLU gate-multiply -- see
    // docs/llama_fpga_plan.md change C; 000-011 bit-identical to the old
    // 2-bit encoding, just zero-extended).
    input  logic [2:0]  vecmul_scale_mode,
    input  logic en_bb_valid,
    input  logic bb_bypass_ln,
    input  logic bb_shift_mode,     // 0: [23:8] (Softmax), 1: [19:4] (RMSNorm)
    input  logic bb_mode_pre_ln,
    input  logic [1:0]  bb_mode_post_ln,
    input  logic bb_ln_pre_scaled,  // 0: ln input has FRAC_W frac bits, 1: FRAC_W+2 (RMSNorm mean)
    input  logic sym_mode_en,
    input  logic alpha_flat_mode,   // true alpha=1 sigmoid slope (SwiGLU's SiLU), see alpha_stabilizer.sv
    output logic vpu_bb_valid_out,
    output logic vpu_vecmul_valid_out,
    output logic vpu_mac_valid_out,
    output logic vpu_max_valid_out
);

    // =========================================================
    // INTERNAL ROUTING WIRES
    // =========================================================
    // Modifiers
    logic signed [W_VEC-1:0] scalar_sub_out [N];
    logic signed [W_VEC-1:0] alpha_stab_out [N];
    logic signed [W_VEC-1:0] sym_mod_out [N];

    // Backbone & ROM
    logic signed [W_MAC-1:0]     bb_mux_out [N]; // Output from the MUX
    logic signed [W_MAC-1:0]     bb_in [N];
    logic signed [W_VEC-1:0]     bb_log [N];
    logic        [LUT_IDX_W-1:0] bb_a_idx [N];
    logic        [W_VEC-1:0]     bb_mantisa [N];
    logic                        bb_is_zero [N];
    logic        [W_VEC-1:0]     rom_e_a [N];

    // Fractional bits of shared_lut_rom.sv's anchor entries (Q0.8, all formats).
    localparam int ANCHOR_Q_BITS = 8;

    // RMSNorm mean of squares into the ln backbone: mk = round(S >> (FRAC_W + log2N - RMS_K)),
    // i.e. the mean at FRAC_W+RMS_K fractional bits, saturated at the W_MAC-bit unsigned max instead
    // of wrapping (a wrapped huge S looked like a tiny one, so the outlier passed through un-normalized).
    localparam int S_SAT_MAX = (1 << W_MAC) - 1;
    logic [5:0]              rms_sh;
    logic signed [ACC_W:0]   rms_s_shifted;
    logic        [W_MAC-1:0] rms_s_sat;
    always_comb begin
        rms_sh        = 6'(FRAC_W) + 6'(rms_log2n) - 6'(RMS_K);
        rms_s_shifted = ($signed({vpu_dot_out[ACC_W-1], vpu_dot_out}) + ((rms_sh == 0) ? 0 : (1 <<< (rms_sh - 1)))) >>> rms_sh;
        rms_s_sat     = (rms_s_shifted > S_SAT_MAX) ? '1 : W_MAC'(rms_s_shifted);
    end

    // Math Engines
    logic        [W_VEC-1:0] mac_in_a [N], mac_in_b [N];
    logic        [W_VEC-1:0] vecmul_in_a [N];
    logic        [INV_W-1:0] vecmul_in_b [N];
    logic        [W_VEC-1:0] vecmul_b8   [N];   // narrow b sources, extended per op_mode below
    logic signed [ACC_W-1:0] vecmul_out_acc [N];
    logic signed [W_VEC-1:0] vecmul_out_trunc [N]; // Truncated to W_VEC for SRAM

    logic int_bb_valid;
    logic int_mac_valid;
    logic int_vecmul_valid;
    logic int_max_valid;

    // =========================================================
    // CROSSBAR MUX NETWORK (Software-Defined Routing)
    // =========================================================

    always_comb begin
        for (int i = 0; i < N; i++) begin
            
            // MUX 1: TR-Backbone Input (Needs padding W_VEC -> W_MAC)
            case (mux_bb_in_sel)
                // RMSNorm's shift is FRAC_W (sum of squares, unrelated to the
                // anchor table). Softmax's dot-product is e_a*mantissa-scaled
                // (2**(ANCHOR_Q_BITS+FRAC_W)), so >>>ANCHOR_Q_BITS restores
                // FRAC_W scale for the ln backbone (== the old [23:8] slice
                // at ANCHOR_Q_BITS=8, without baking the 8 in).
                3'b000: bb_mux_out[i] = bb_shift_mode ? rms_s_sat : W_MAC'(vpu_dot_out >>> ANCHOR_Q_BITS);
                3'b001: bb_mux_out[i] = W_MAC'(alpha_stab_out[i]);   // GELU stabilized
                3'b010: bb_mux_out[i] = W_MAC'(scalar_sub_out[i]);   // Softmax (x - max)
                3'b011: bb_mux_out[i] = W_MAC'(sram_data_a[i]);      // Raw pass-through
                3'b100: bb_mux_out[i] = W_MAC'(ctrl_scalar_sub_val); // Direct Scalar Injection
                default: bb_mux_out[i] = '0;
            endcase

            // MUX 2: VPU MAC Input A
            case (mux_mac_a_sel)
                2'b00: mac_in_a[i] = sram_data_a[i];           // Standard (x_i)
                2'b01: mac_in_a[i] = bb_is_zero[i] ? 8'd128 : $signed(rom_e_a[i]); // Softmax/GELU (e_a) -> Explicit Cast
                2'b10: mac_in_a[i] = scalar_sub_out[i];        // Variance (x - mu)
                default: mac_in_a[i] = sram_data_a[i];
            endcase

            // MUX 3: VPU MAC Input B
            case (mux_mac_b_sel)
                2'b00: mac_in_b[i] = sram_data_b[i];            // Standard (x_i)
                2'b01: mac_in_b[i] = bb_is_zero[i] ? ($signed(bb_mantisa[i]) << 1) : $signed(bb_mantisa[i]); // Softmax/GELU (e_frac) -> Explicit Cast
                2'b10: mac_in_b[i] = sram_data_a[i];
                default: mac_in_b[i] = sram_data_b[i];
            endcase

            // MUX 4: VecMul Input A
            case (mux_vecmul_a_sel)
                2'b00: vecmul_in_a[i] = sram_data_a[i];        // Standard (x_i or e^x)
                2'b01: vecmul_in_a[i] = scalar_sub_out[i];     // RMSNorm (x - mu)
                2'b10: vecmul_in_a[i] = $signed(bb_mantisa[0]);// Broadcast TR-scalar + Cast
                2'b11: vecmul_in_a[i] = bb_is_zero[i] ? 8'd128 : $signed(rom_e_a[i]);
            endcase

            // MUX 5: VecMul Input B
            case (mux_vecmul_b_sel)
                2'b00: vecmul_b8[i] = sram_data_b[i];         // Standard
                2'b01: vecmul_b8[i] = sym_mod_out[i];         // GELU Symmetry (sigma)
                2'b10: vecmul_b8[i] = bb_is_zero[i] ? ($signed(bb_mantisa[i]) << 1) : $signed(bb_mantisa[i]);
                default: vecmul_b8[i] = sram_data_b[i];
            endcase

            // Widen the b operand for the engine: SS mode sign-extends the 8-bit sources, SU/UU zero-extend
            // (as the engine did when b was 8 bits). 2'b11 injects RMSNorm's wide unsigned inv_rms.
            if (mux_vecmul_b_sel == 2'b11) vecmul_in_b[i] = inv_rms_in;
            else if (vecmul_op_mode == 2'd0) vecmul_in_b[i] = INV_W'($signed(vecmul_b8[i]));
            else                             vecmul_in_b[i] = INV_W'(vecmul_b8[i]);
        end
    end

    // =========================================================
    // VPU Backbone Direct Wire
    // =========================================================
    always_comb begin
        for(int i=0; i<N; i++) begin
            bb_in[i] = bb_mux_out[i];
        end
    end

    // =========================================================
    // MUX 6: VPU Final Output
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(int i=0; i<N; i++) vpu_data_out[i] <= '0;
        end else begin
            for(int i=0; i<N; i++) begin
                case (mux_vpu_out_sel)
                    3'b000: vpu_data_out[i] <= vecmul_out_trunc[i];
                    3'b001: vpu_data_out[i] <= bb_log[i]; 
                    3'b010: vpu_data_out[i] <= alpha_stab_out[i];
                    3'b011: vpu_data_out[i] <= scalar_sub_out[i];
                    default: vpu_data_out[i] <= vecmul_out_trunc[i];
                endcase
            end
        end
    end

    // =========================================================
    // SUBMODULE INSTANTIATIONS
    // =========================================================
    piped_max #(
        .NUM_INPUTS(N), .DATA_WIDTH(W_VEC)
    ) u_max (
        .clk(clk), .rst_n(rst_n), .valid_in(en_piped_max),
        .in_data(sram_data_a), .max_out(vpu_max_out), .valid_out(int_max_valid)
    );

    scalar_sub #(
        .NUM_INPUTS(N), .DATA_WIDTH(W_VEC)
    ) u_sub (
        .in_data(sram_data_a), .sub_val(ctrl_scalar_sub_val), .out_data(scalar_sub_out)
    );

    alpha_stabilizer #(
        .N(N), .W(W_VEC), .FRAC_W(FRAC_W)
    ) u_alpha (
        .in_vec(sram_data_a), .flat_mode(alpha_flat_mode), .out_vec(alpha_stab_out)
    );

    symmetry_modifier #(
        .N(N), .WIDTH_X(W_VEC), .WIDTH_Y(W_VEC), .FRAC_W(FRAC_W)
    ) u_sym (
        .x_raw(sram_data_a), .y_sig(sram_data_b), .mode_en(sym_mode_en), .sig_corrected(sym_mod_out)
    );

    logic        [7:0]             rom_e_a_8bit [N];

    tr_backbone_wrapper #(
        .N(N), .WIDTH_IN(W_MAC), .WIDTH_OUT(W_VEC), .FRAC_W(FRAC_W), .LUT_IDX_W(LUT_IDX_W)
    ) u_backbone (
        .clk(clk), .rst_n(rst_n), .in_valid(en_bb_valid), .out_valid(int_bb_valid),
        .bypass_ln(bb_bypass_ln), .mode_pre_ln(bb_mode_pre_ln), .mode_post_ln(bb_mode_post_ln), .ln_pre_scaled(bb_ln_pre_scaled),
        .vec_in(bb_in), // Conditionally shifted input!
        .log_out(bb_log), .a_idx_out(bb_a_idx), .mantisa_out(bb_mantisa), .is_zero_out(bb_is_zero)
    );

    shared_lut_rom #(.N(N), .LUT_IDX_W(LUT_IDX_W)) u_rom (.a_idx(bb_a_idx), .e_a(rom_e_a_8bit));

    always_comb begin
        for (int i = 0; i < N; i++) begin
            rom_e_a[i] = W_VEC'(rom_e_a_8bit[i]);
        end
    end

    mac_array_engine #(
        .N(N), .W(W_VEC), .ACC_W(ACC_W)
    ) u_mac (
        .clk(clk), .rst_n(rst_n), .in_valid(en_mac_valid), .in_ready(),
        .op_mode(mac_op_mode), .a(mac_in_a), .b(mac_in_b), .c('0),
        .clear_acc(mac_clear_acc), .out_valid(int_mac_valid), .out_ready(1'b1), .out_dot(vpu_dot_out)
    );

    vec_mul_array_engine #(
        .N(N), .W(W_VEC), .WB(INV_W), .ACC_W(ACC_W)
    ) u_vecmul (
        .clk(clk), .rst_n(rst_n), .in_valid(en_vecmul_valid), .in_ready(),
        .op_mode(vecmul_op_mode), .a(vecmul_in_a), .b(vecmul_in_b),
        .out_valid(int_vecmul_valid), .out_ready(1'b1), .out_vec(vecmul_out_acc)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vpu_bb_valid_out     <= 1'b0;
            vpu_mac_valid_out    <= 1'b0;
            vpu_vecmul_valid_out <= 1'b0;
            vpu_max_valid_out    <= 1'b0;
        end else begin
            vpu_bb_valid_out     <= int_bb_valid;
            vpu_mac_valid_out    <= int_mac_valid;
            vpu_vecmul_valid_out <= int_vecmul_valid;
            vpu_max_valid_out    <= int_max_valid;
        end
    end

    assign vpu_exp_raw = vecmul_out_acc[0][15:0];

    localparam signed [W_VEC-1:0] VECMUL_SAT_MAX = (1 <<< (W_VEC-1)) - 1;
    localparam signed [W_VEC-1:0] VECMUL_SAT_MIN = -(1 <<< (W_VEC-1));

    always_comb begin
        for(int i=0; i<N; i++) begin
            logic signed [ACC_W-1:0] shifted;
            if (vecmul_scale_mode == 3'b011) begin
                // Softmax Q0.8 UNSIGNED probability output: shift =
                // anchor_q_bits(8, fixed for every FRAC_W<=8 format) +
                // FRAC_W - 8 = FRAC_W. Saturate at 255 (~0.996) instead of
                // wrapping (e.g. the single-dominant-logit case can land
                // exactly on 256).
                shifted = vecmul_out_acc[i] >>> FRAC_W;
                vecmul_out_trunc[i] = (shifted > 255) ? W_VEC'(255) : W_VEC'(shifted);
            end else if (vecmul_scale_mode == 3'b100) begin
                // SIGNED saturating variant of mode 001 (same FRAC_W shift),
                // for SwiGLU's final gate-multiply (silu*g): that product
                // routinely exceeds W_VEC's representable range (unlike
                // GELU's/RMSNorm's own single-operand-scale products, which
                // rarely do), and wrapping there is a correctness risk, not
                // just added noise -- see docs/llama_fpga_plan.md's
                // wraparound finding. Mirrors mode 011's saturating
                // precedent, signed instead of unsigned.
                shifted = vecmul_out_acc[i] >>> FRAC_W;
                if (shifted > VECMUL_SAT_MAX) vecmul_out_trunc[i] = VECMUL_SAT_MAX;
                else if (shifted < VECMUL_SAT_MIN) vecmul_out_trunc[i] = VECMUL_SAT_MIN;
                else vecmul_out_trunc[i] = W_VEC'(shifted);
            end else if (vecmul_scale_mode == 3'b101) begin
                // RMSNorm apply: x * inv_rms (inv_rms has FRAC_W+4 fractional bits) -> Q.FRAC_W, ROUNDED,
                // signed saturate. Rounding adds half an output LSB before the shift.
                shifted = (vecmul_out_acc[i] + (1 <<< (FRAC_W + 3))) >>> (FRAC_W + 4);
                if (shifted > VECMUL_SAT_MAX) vecmul_out_trunc[i] = VECMUL_SAT_MAX;
                else if (shifted < VECMUL_SAT_MIN) vecmul_out_trunc[i] = VECMUL_SAT_MIN;
                else vecmul_out_trunc[i] = W_VEC'(shifted);
            end else if (vecmul_scale_mode == 3'b010) begin
                // anchor_q_bits fixed at 8 -- not FRAC_W-dependent.
                vecmul_out_trunc[i] = W_VEC'(vecmul_out_acc[i] >>> 8);
            end else if (vecmul_scale_mode == 3'b001) begin
                vecmul_out_trunc[i] = W_VEC'(vecmul_out_acc[i] >>> FRAC_W);
            end else begin
                vecmul_out_trunc[i] = W_VEC'(vecmul_out_acc[i]);
            end
        end
    end

endmodule