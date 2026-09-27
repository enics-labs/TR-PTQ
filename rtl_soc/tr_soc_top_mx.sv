`timescale 1ns/1ps

/*
 * @module   tr_soc_top_mx
 * @brief    Top-level MX-datapath SoC integration -- the MX counterpart of
 *           tr_soc_top_int, using shared-exponent (MX) formatting instead of
 *           a fixed-point requantizer between the linear and nonlinear
 *           engines.
 * @details  Instantiates: tr_soc_ctrl_mx (master FSM), tr_matmul_ctrl
 *           (streaming matmul sequencer sharing u_dot/u_req_mx via the same
 *           busy-gated mux as the INT top), dot_product_engine (M-lane
 *           matrix-vector MAC), requantize_engine_mx (compresses the wide
 *           accumulator to OUT_W mantissas plus a combined mx_shared_exp
 *           from a_mat_exp/b_vec_exp), dynamic_shifter_mx (expands the
 *           requantized mantissas up to the wider VPU_W domain by
 *           mx_shared_exp before the VPU consumes them, registered here to
 *           break a timing-critical path), tr_nonlinear_vpu (running at
 *           VPU_W for both its vector and internal-backbone width, since MX
 *           carries no separate narrow SRAM format), and formatter_mx
 *           (compresses the VPU's VPU_W-wide result back down to W-bit
 *           mantissas plus a shared exponent for output/SRAM).
 *
 * @param    M      Parallel output lanes of the linear engine (matmul/dot tile height).
 * @param    N      Vector dimension of the nonlinear VPU / contraction tile width.
 * @param    W      Narrow MX mantissa width (a_mat/b_vec/ext_sram_b/vpu_data_out).
 * @param    ACC_W  Accumulator width (c_vec bias, dot_acc_out, vpu_dot_out).
 * @param    VPU_W  Wide internal width the VPU and its backbone operate at
 *                    (both W_VEC and W_MAC of tr_nonlinear_vpu here).
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
    input  logic clear_acc,   // streaming matmul: 1=init accumulator, 0=accumulate

    // Non-Linear Engine Memory Interfaces
    input  logic signed [W-1:0]     ext_sram_b [N],
    output logic vpu_out_valid,
    output logic signed [W-1:0]     vpu_data_out [N],
    output logic signed [7:0]       vpu_data_exp,

    // Streaming matmul memory interface (CMD=0x04).  The wrapper holds the
    // MX exponents (a_mat_exp, b_vec_exp) constant across the contraction, so
    // the result is {mm_out_data mantissas[M], mm_out_exp shared exponent}.
    output logic [15:0]             mm_tile_row,
    output logic [15:0]             mm_tile_col,
    output logic                    mm_mem_rd,
    input  logic                    mm_mem_valid,   // a_tile/b_tile valid this cycle
    input  logic [W-1:0]            mm_a_tile [M][N],
    input  logic [W-1:0]            mm_b_tile [N],
    output logic                    mm_out_we,
    output logic [15:0]             mm_out_row,
    output logic signed [W-1:0]     mm_out_data [M],
    output logic signed [7:0]       mm_out_exp
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

    // Streaming matmul controller <-> datapath
    logic                           mm_start, mm_done, mm_busy;
    logic [15:0]                    mm_num_rt, mm_num_ct;
    logic                           mm_dot_in_valid, mm_clear_acc;
    logic [W-1:0]                   dot_a_mux [M][N];
    logic [W-1:0]                   dot_b_mux [N];
    logic signed [ACC_W-1:0]        dot_c_mux [M];
    logic                           dot_iv_mux, dot_clr_mux;

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
        .vpu_bb_valid(vpu_bb_valid), .vpu_vecmul_valid(vpu_vecmul_valid), .vpu_mac_valid(vpu_mac_valid),
        .mm_start(mm_start), .mm_num_row_tiles(mm_num_rt), .mm_num_ctiles(mm_num_ct), .mm_done(mm_done)
    );

    // =========================================================
    // 1b. STREAMING MATMUL SEQUENCER (shares u_dot + u_req_mx)
    // =========================================================
    tr_matmul_ctrl #(.M(M), .N(N)) u_mm (
        .clk(clk), .rst_n(rst_n),
        .start(mm_start), .num_row_tiles(mm_num_rt), .num_ctiles(mm_num_ct),
        .busy(mm_busy), .done(mm_done),
        .tile_row(mm_tile_row), .tile_col(mm_tile_col), .mem_rd(mm_mem_rd),
        .mem_valid(mm_mem_valid),
        .dot_in_valid(mm_dot_in_valid), .clear_acc(mm_clear_acc),
        .req_valid(req_out_valid),
        .result_we(mm_out_we), .result_row(mm_out_row)
    );
    assign mm_out_data = req_vec_out;    // mantissas on the capture cycle
    assign mm_out_exp  = mx_shared_exp;  // shared exponent for the row-tile

    // Route the linear engine to the matmul controller while it is busy,
    // otherwise to the legacy single-tile ports.
    always_comb begin
        if (mm_busy) begin
            dot_iv_mux  = mm_dot_in_valid;
            dot_clr_mux = mm_clear_acc;
            for (int m = 0; m < M; m++) begin
                dot_c_mux[m] = '0;
                for (int i = 0; i < N; i++) dot_a_mux[m][i] = mm_a_tile[m][i];
            end
            for (int i = 0; i < N; i++) dot_b_mux[i] = mm_b_tile[i];
        end else begin
            dot_iv_mux  = dot_in_valid;
            dot_clr_mux = clear_acc;
            dot_a_mux   = a_mat;
            dot_b_mux   = b_vec;
            dot_c_mux   = c_vec;
        end
    end

    // =========================================================
    // 2. LINEAR DOT ENGINE
    // =========================================================
    dot_product_engine #(
        .M(M), .N(N), .W(W), .ACC_W(ACC_W)
    ) u_dot (
        .clk(clk), .rst_n(rst_n),
        .in_valid(dot_iv_mux), .in_ready(dot_in_ready),
        .op_mode(2'b00), .a_mat(dot_a_mux), .b_vec(dot_b_mux), .c_vec(dot_c_mux),
        .clear_acc(dot_clr_mux), .out_valid(dot_out_valid), .out_ready(req_in_ready),
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
    
    // Extrapolate the valid M items into N items, filling with zeros
    logic signed [W-1:0] padded_req_vec_out [N];
    always_comb begin
        for(int i = 0; i < N; i++) begin
            if (i < M) padded_req_vec_out[i] = req_vec_out[i];
            else       padded_req_vec_out[i] = '0;
        end
    end

    // =========================================================
    // 3b. PRE-COMPUTED PIPELINE SHIFTER (Fixes -500ps Violation)
    // =========================================================
    logic signed [VPU_W-1:0] shifter_out_comb [N];
    logic signed [VPU_W-1:0] shifted_req_vec_out_reg [N];
    
    dynamic_shifter_mx #(
        .N(N), .IN_W(W), .OUT_W(VPU_W)
    ) u_top_shifter (
        .data_in(padded_req_vec_out),
        .shift_amount(mx_shared_exp),
        .shift_dir(1'b1), // Expand
        .data_out(shifter_out_comb)
    );

    // Register the shifted result to break the critical path
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(int i=0; i<N; i++) shifted_req_vec_out_reg[i] <= '0;
        end else begin
            shifted_req_vec_out_reg <= shifter_out_comb;
        end
    end

    // =========================================================
    // 4. VPU INPUT ROUTING
    // =========================================================
    always_comb begin
        for(int i = 0; i < N; i++) begin
            automatic logic signed [VPU_W-1:0] sram_a_route;
            
            // FSM now just drives a fast MUX, shaving ~600ps off the critical path
            sram_a_route = ctrl_enable_linear_shift ? shifted_req_vec_out_reg[i] : VPU_W'(padded_req_vec_out[i]);

            vpu_sram_a_in[i] = (src_sram_a_sel) ? ctrl_scratch_a[i] : sram_a_route;
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