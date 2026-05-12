`timescale 1ns/1ps

/*
 * @module   tr_soc_top_mx
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    M               TODO: Add description
 * @param    N               TODO: Add description
 * @param    W               TODO: Add description
 * @param    ACC_W           TODO: Add description
 * @param    VPU_W           TODO: Add description
 */
module tr_soc_top_mx #(
    parameter int M        = 4,
    parameter int N        = 8,
    parameter int W        = 8,
    parameter int ACC_W    = 32,
    parameter int VPU_W    = 16
)(
    input  logic clk,
    input  logic rst_n,

    // RISC-V MMIO Interface
    input  logic [7:0]              mmio_addr,
    input  logic [31:0]             mmio_wdata,
    input  logic mmio_wen,
    output logic [31:0]             mmio_rdata,

    // Linear Engine External Inputs + MX Exponents
    input  logic dot_in_valid,
    output logic dot_in_ready,
    input  logic [W-1:0]            a_mat [M][N],
    input  logic signed [7:0]       a_mat_exp,
    input  logic [W-1:0]            b_vec [N],
    input  logic signed [7:0]       b_vec_exp,
    input  logic signed [ACC_W-1:0] c_vec [M],
    
    // Non-Linear Engine Memory Interfaces
    input  logic signed [W-1:0]     ext_sram_b [N],
    output logic vpu_out_valid,
    output logic signed [W-1:0]     vpu_data_out [N],
    output logic signed [7:0]       vpu_data_exp
);

    // =========================================================
    // Interconnect Wires
    // =========================================================
    logic                           dot_out_valid;
    logic                           req_in_ready;
    assign req_in_ready = 1'b1; // Requantize engine does not exert backpressure
    logic signed [ACC_W-1:0]        dot_acc_out [M];
    
    logic                           req_out_valid;
    logic signed [W-1:0]            req_vec_out [M];
    logic signed [7:0]              mx_shared_exp;

    logic signed [VPU_W-1:0]             vpu_sram_a_in [N];
    logic signed [VPU_W-1:0]             vpu_sram_b_in [N];
    
    // VPU Controller -> Datapath Wires
    logic                           ctrl_enable_linear_shift;
    logic [2:0]                     mux_bb_in_sel;
    logic [1:0]                     mux_mac_a_sel, mux_mac_b_sel;
    logic [1:0]                     mux_vecmul_a_sel, mux_vecmul_b_sel;
    logic [2:0]                     mux_vpu_out_sel;
    logic                           en_piped_max, en_mac_valid, en_vecmul_valid, en_bb_valid;
    logic                           mac_clear_acc, bb_shift_mode, bb_bypass_ln, bb_mode_pre_ln, sym_mode_en;
    logic [1:0]                     mac_op_mode, vecmul_op_mode, vecmul_scale_mode, bb_mode_post_ln;
    logic signed [VPU_W-1:0]             ctrl_scalar_sub_val;
    logic                           src_sram_a_sel, src_sram_b_sel, write_ext_sram;
    logic signed [VPU_W-1:0]             ctrl_scratch_a [N];
    logic signed [VPU_W-1:0]             ctrl_scratch_b [N];

    // VPU Outputs
    logic signed [VPU_W-1:0]             vpu_max_out;
    logic signed [ACC_W-1:0]        vpu_dot_out;
    logic signed [VPU_W-1:0]             vpu_raw_out [N];
    logic                           vpu_bb_valid, vpu_vecmul_valid, vpu_mac_valid;

    // =========================================================
    // 1. MX MASTER CONTROLLER
    // =========================================================
    tr_soc_ctrl_mx #(
        .N(N), .W(VPU_W), .ACC_W(ACC_W)
    ) u_ctrl (
        .clk(clk), .rst_n(rst_n),
        .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata), .mmio_wen(mmio_wen), .mmio_rdata(mmio_rdata),
        .ctrl_enable_linear_shift(ctrl_enable_linear_shift),
        
        .mux_bb_in_sel(mux_bb_in_sel), .mux_mac_a_sel(mux_mac_a_sel), .mux_mac_b_sel(mux_mac_b_sel),
        .mux_vecmul_a_sel(mux_vecmul_a_sel), .mux_vecmul_b_sel(mux_vecmul_b_sel), .mux_vpu_out_sel(mux_vpu_out_sel),
        .en_piped_max(en_piped_max), .en_mac_valid(en_mac_valid), .en_vecmul_valid(en_vecmul_valid), .en_bb_valid(en_bb_valid),
        .mac_clear_acc(mac_clear_acc), .mac_op_mode(mac_op_mode), .vecmul_op_mode(vecmul_op_mode), .vecmul_scale_mode(vecmul_scale_mode),
        .bb_shift_mode(bb_shift_mode), .bb_bypass_ln(bb_bypass_ln), .bb_mode_pre_ln(bb_mode_pre_ln),
        .bb_mode_post_ln(bb_mode_post_ln), .sym_mode_en(sym_mode_en), .ctrl_scalar_sub_val(ctrl_scalar_sub_val),
        .src_sram_a_sel(src_sram_a_sel), .src_sram_b_sel(src_sram_b_sel), .write_ext_sram(write_ext_sram),
        .scratch_a_out(ctrl_scratch_a), .scratch_b_out(ctrl_scratch_b),
        
        .vpu_data_out(vpu_raw_out), .vpu_max_out(vpu_max_out), .vpu_dot_out(vpu_dot_out),
        .vpu_bb_valid(vpu_bb_valid), .vpu_vecmul_valid(vpu_vecmul_valid), .vpu_mac_valid(vpu_mac_valid)
    );

    // =========================================================
    // 2. LINEAR DOT ENGINE
    // =========================================================
    dot_product_engine #(
        .M(M), .N(N), .W(W), .ACC_W(ACC_W)
    ) u_dot (
        .clk(clk), .rst_n(rst_n),
        .in_valid(dot_in_valid), .in_ready(dot_in_ready),
        .op_mode(2'b00), .a_mat(a_mat), .b_vec(b_vec), .c_vec(c_vec),
        .clear_acc(1'b1), .out_valid(dot_out_valid), .out_ready(req_in_ready),
        .out_vec(dot_acc_out)
    );

    // =========================================================
    // 3. MX REQUANTIZER ENGINE
    // =========================================================
    requantize_engine_mx #(
        .N(M), .ACC_W(ACC_W), .OUT_W(W)
    ) u_req_mx (
        .clk(clk), .rst_n(rst_n),
        .dot_in_valid(dot_out_valid),
        .dot_in(dot_acc_out),
        .exp_act_in(a_mat_exp),
        .exp_weight_in(b_vec_exp),
        .req_vec_out(req_vec_out),
        .exp_total_out(mx_shared_exp),
        .req_out_valid(req_out_valid)
    );

    logic signed [VPU_W-1:0] shifted_req_vec_out [N];
    
    // Extrapolate the valid M items into N items, filling with zeros
    logic signed [W-1:0] padded_req_vec_out [N];
    always_comb begin
        for(int i = 0; i < N; i++) begin
            if (i < M) padded_req_vec_out[i] = req_vec_out[i];
            else       padded_req_vec_out[i] = '0;
        end
    end

    dynamic_shifter_mx #(
        .N(N), .IN_W(W), .OUT_W(VPU_W)
    ) u_top_shifter (
        .data_in(padded_req_vec_out),
        .shift_amount(ctrl_enable_linear_shift ? mx_shared_exp : 8'd0),
        .shift_dir(1'b1), // Expand
        .data_out(shifted_req_vec_out)
    );

    // =========================================================
    // 4. VPU INPUT ROUTING
    // =========================================================
    always_comb begin
        for(int i = 0; i < N; i++) begin
            vpu_sram_a_in[i] = (src_sram_a_sel) ? ctrl_scratch_a[i] : shifted_req_vec_out[i];
            vpu_sram_b_in[i] = (src_sram_b_sel) ? ctrl_scratch_b[i] : VPU_W'(ext_sram_b[i]);
        end
    end

    // =========================================================
    // 5. SHARED NON-LINEAR VPU
    // =========================================================
    tr_nonlinear_vpu #(
        .N(N),
        .W_VEC(VPU_W),
        .W_MAC(VPU_W),
        .ACC_W(ACC_W),
        .FRAC_W(4),
        .LUT_IDX_W(3)
    ) u_vpu (
        .clk(clk), .rst_n(rst_n),
        .sram_data_a(vpu_sram_a_in), .sram_data_b(vpu_sram_b_in), .vpu_data_out(vpu_raw_out),
        
        .ctrl_scalar_sub_val(ctrl_scalar_sub_val), .vpu_max_out(vpu_max_out), .vpu_dot_out(vpu_dot_out),
        .mux_bb_in_sel(mux_bb_in_sel), .mux_mac_a_sel(mux_mac_a_sel), .mux_mac_b_sel(mux_mac_b_sel),
        .mux_vecmul_a_sel(mux_vecmul_a_sel), .mux_vecmul_b_sel(mux_vecmul_b_sel), .mux_vpu_out_sel(mux_vpu_out_sel),
        .en_piped_max(en_piped_max), .en_mac_valid(en_mac_valid), .en_vecmul_valid(en_vecmul_valid), .en_bb_valid(en_bb_valid),
        .mac_clear_acc(mac_clear_acc), .mac_op_mode(mac_op_mode), .vecmul_op_mode(vecmul_op_mode), .vecmul_scale_mode(vecmul_scale_mode),
        .bb_shift_mode(bb_shift_mode), .bb_bypass_ln(bb_bypass_ln), .bb_mode_pre_ln(bb_mode_pre_ln),
        .bb_mode_post_ln(bb_mode_post_ln), .sym_mode_en(sym_mode_en),
        .vpu_bb_valid_out(vpu_bb_valid), .vpu_vecmul_valid_out(vpu_vecmul_valid), .vpu_mac_valid_out(vpu_mac_valid)
    );

    // =========================================================
    // 6. THE MX FORMATTER (Final Compression to SRAM)
    // =========================================================
    formatter_mx #(
        .N(N), .VPU_W(VPU_W), .MX_W(W)
    ) u_formatter (
        .clk(clk), .rst_n(rst_n),
        .valid_in(write_ext_sram),
        .vpu_data_in(vpu_raw_out),
        .valid_out(vpu_out_valid),
        .mx_mantissas(vpu_data_out),
        .mx_shared_exp(vpu_data_exp)
    );

endmodule