/*
 * @module   tr_nonlinear_vpu
 * @brief    Software-Defined Vector Processing Unit.
 * @details  A strictly combinational/pipelined crossbar that routes SRAM data 
 *           through isolated math modifiers, a log-domain backbone, and local 
 *           vector/MAC engines. Contains ZERO internal state memory.
 */
module tr_nonlinear_vpu #(
    parameter int N         = 8,
    parameter int W_VEC     = 8,
    parameter int W_MAC     = 16,
    parameter int ACC_W     = 32,
    parameter int FRAC_W    = 4,
    parameter int LUT_IDX_W = 3
)(
    input  logic clk,
    input  logic rst_n,

    // =========================================================
    // 1. External Memory Interfaces (SRAM)
    // =========================================================
    input  logic signed [W_VEC-1:0] sram_data_a [N],
    input  logic signed [W_VEC-1:0] sram_data_b [N],
    output logic signed [W_VEC-1:0] vpu_data_out [N],
    
    // =========================================================
    // 2. Controller Register Interface (Broadcast Scalars)
    // =========================================================
    input  logic signed [W_VEC-1:0] ctrl_scalar_sub_val, // For x - max or x - mu
    output logic signed [W_VEC-1:0] vpu_max_out,         // To Controller Reg
    output logic signed [ACC_W-1:0] vpu_dot_out,         // To Controller Reg

    // =========================================================
    // 3. Crossbar Routing Controls
    // =========================================================
    input  logic [1:0]  mux_bb_in_sel,   // Backbone Input MUX
    input  logic [1:0]  mux_mac_a_sel,   // MAC Input A MUX
    input  logic        mux_mac_b_sel,   // MAC Input B MUX
    input  logic [1:0]  mux_vecmul_a_sel,// VecMul Input A MUX
    input  logic        mux_vecmul_b_sel,// VecMul Input B MUX
    input  logic [2:0]  mux_vpu_out_sel, // Final Output MUX to SRAM

    // =========================================================
    // 4. Submodule Enables & Modes
    // =========================================================
    input  logic        en_piped_max,
    input  logic        en_mac_valid,
    input  logic        en_vecmul_valid,
    input  logic        mac_clear_acc,
    input  logic [1:0]  mac_op_mode,
    input  logic [1:0]  vecmul_op_mode,
    
    input  logic        bb_mode_pre_ln,
    input  logic [1:0]  bb_mode_post_ln,
    input  logic        sym_mode_en
);

    // =========================================================
    // INTERNAL ROUTING WIRES
    // =========================================================
    // Modifiers
    logic signed [W_VEC-1:0] scalar_sub_out [N];
    logic signed [W_VEC-1:0] alpha_stab_out [N];
    logic signed [W_VEC-1:0] sym_mod_out [N];

    // Backbone & ROM
    logic signed [W_MAC-1:0] bb_in [N];
    logic [LUT_IDX_W-1:0]    bb_a_idx [N];
    logic [W_VEC-1:0]        bb_mantisa [N];
    logic [W_VEC-1:0]        rom_e_a [N];

    // Math Engines
    logic [W_VEC-1:0]        mac_in_a [N], mac_in_b [N];
    logic [W_VEC-1:0]        vecmul_in_a [N], vecmul_in_b [N];
    logic signed [ACC_W-1:0] vecmul_out_acc [N];
    logic signed [W_VEC-1:0] vecmul_out_trunc [N]; // Truncated to W_VEC for SRAM

    // =========================================================
    // CROSSBAR MUX NETWORK (Software-Defined Routing)
    // =========================================================

    always_comb begin
        for (int i = 0; i < N; i++) begin
            
            // MUX 1: TR-Backbone Input (Needs padding W_VEC -> W_MAC)
            case (mux_bb_in_sel)
                2'b00: bb_in[i] = W_MAC'(vpu_dot_out);         // Broadcast Scalar
                2'b01: bb_in[i] = W_MAC'(alpha_stab_out[i]);   // GELU stabilized
                2'b10: bb_in[i] = W_MAC'(scalar_sub_out[i]);   // Softmax (x - max)
                2'b11: bb_in[i] = W_MAC'(sram_data_a[i]);      // Raw pass-through
                default: bb_in[i] = W_MAC'(sram_data_a[i]);
            endcase

            // MUX 2: VPU MAC Input A
            case (mux_mac_a_sel)
                2'b00: mac_in_a[i] = sram_data_a[i];           // Standard (x_i)
                2'b01: mac_in_a[i] = $signed(rom_e_a[i]);      // Softmax/GELU (e_a) -> Explicit Cast
                2'b10: mac_in_a[i] = scalar_sub_out[i];        // Variance (x - mu)
                default: mac_in_a[i] = sram_data_a[i];
            endcase

            // MUX 3: VPU MAC Input B
            case (mux_mac_b_sel)
                1'b0: mac_in_b[i] = sram_data_b[i];            // Standard (x_i)
                1'b1: mac_in_b[i] = $signed(bb_mantisa[i]);    // Softmax/GELU (e_frac) -> Explicit Cast
            endcase

            // MUX 4: VecMul Input A
            case (mux_vecmul_a_sel)
                2'b00: vecmul_in_a[i] = sram_data_a[i];        // Standard (x_i or e^x)
                2'b01: vecmul_in_a[i] = scalar_sub_out[i];     // RMSNorm (x - mu)
                2'b10: vecmul_in_a[i] = $signed(bb_mantisa[0]);// Broadcast TR-scalar + Cast
                default: vecmul_in_a[i] = sram_data_a[i];
            endcase

            // MUX 5: VecMul Input B
            case (mux_vecmul_b_sel)
                1'b0: vecmul_in_b[i] = sram_data_b[i];         // Standard
                1'b1: vecmul_in_b[i] = sym_mod_out[i];         // GELU Symmetry (sigma)
            endcase

            // MUX 6: VPU Final Output to SRAM
            case (mux_vpu_out_sel)
                3'b000: vpu_data_out[i] = vecmul_out_trunc[i]; 
                3'b001: vpu_data_out[i] = $signed(bb_mantisa[i]); // Vector pass-through -> Explicit Cast
                3'b010: vpu_data_out[i] = alpha_stab_out[i];
                3'b011: vpu_data_out[i] = scalar_sub_out[i];
                default: vpu_data_out[i] = vecmul_out_trunc[i];
            endcase
            
        end
    end

    // =========================================================
    // SUBMODULE INSTANTIATIONS
    // =========================================================

    piped_max #(
        .NUM_INPUTS(N),
        .DATA_WIDTH(W_VEC)
    ) u_max (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(en_piped_max),
        .in_data(sram_data_a),
        .max_out(vpu_max_out),
        .valid_out()
    );

    scalar_sub #(
        .NUM_INPUTS(N),
        .DATA_WIDTH(W_VEC)
    ) u_sub (
        .in_data(sram_data_a),
        .sub_val(ctrl_scalar_sub_val),
        .out_data(scalar_sub_out)
    );

    alpha_stabilizer #(
        .N(N),
        .W(W_VEC)
    ) u_alpha (
        .in_vec(sram_data_a),
        .out_vec(alpha_stab_out)
    );

    symmetry_modifier #(
        .N(N),
        .WIDTH_X(W_VEC),
        .WIDTH_Y(W_VEC),
        .FRAC_W(FRAC_W)
    ) u_sym (
        .x_raw(sram_data_a),
        .y_sig(sram_data_b), 
        .mode_en(sym_mode_en),
        .sig_corrected(sym_mod_out)
    );

    tr_backbone_wrapper #(
        .N(N),
        .WIDTH_IN(W_MAC),
        .WIDTH_OUT(W_VEC), 
        .FRAC_W(FRAC_W),
        .LUT_IDX_W(LUT_IDX_W)
    ) u_backbone (
        .mode_pre_ln(bb_mode_pre_ln),
        .mode_post_ln(bb_mode_post_ln),
        .vec_in(bb_in),
        .a_idx_out(bb_a_idx),
        .mantisa_out(bb_mantisa),
        .is_zero_out()
    );

    shared_lut_rom #(
        .N(N)
    ) u_rom (
        .a_idx(bb_a_idx),
        .e_a(rom_e_a)
    );

    mac_array_engine #(
        .N(N),
        .W(W_VEC),
        .ACC_W(ACC_W)
    ) u_mac (
        .clk(clk), 
        .rst_n(rst_n), 
        .in_valid(en_mac_valid),
        .in_ready(),
        .op_mode(mac_op_mode),
        .a(mac_in_a),
        .b(mac_in_b), 
        .c('0),
        .clear_acc(mac_clear_acc),
        .out_valid(),
        .out_ready(1'b1),
        .out_dot(vpu_dot_out)
    );

    vec_mul_array_engine #(
        .N(N),
        .W(W_VEC),
        .ACC_W(ACC_W)
    ) u_vecmul (
        .clk(clk), 
        .rst_n(rst_n), 
        .in_valid(en_vecmul_valid),
        .in_ready(),
        .op_mode(vecmul_op_mode),
        .a(vecmul_in_a),
        .b(vecmul_in_b),
        .out_valid(),
        .out_ready(1'b1),
        .out_vec(vecmul_out_acc)
    );

    // Cast VecMul ACC output back to base width
    always_comb begin
        for(int i=0; i<N; i++) vecmul_out_trunc[i] = vecmul_out_acc[i][W_VEC-1:0];
    end

endmodule