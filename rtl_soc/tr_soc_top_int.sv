`timescale 1ns/1ps

/*
 * @module   tr_soc_top_int
 * @brief    Top-level INT-datapath SoC integration: wires the matmul/
 *           requantize linear pipeline and the log/exp nonlinear VPU
 *           together under one MMIO-driven controller.
 * @details  Instantiates: tr_soc_ctrl_int (master FSM), tr_matmul_ctrl
 *           (streaming matmul sequencer, sharing the same dot+requant
 *           datapath as the legacy single-tile ports via a busy-gated mux),
 *           dot_product_engine (M-lane matrix-vector MAC), requantize_engine_int
 *           (Acc*M+Bias>>>S rescale back to W-bit), and tr_nonlinear_vpu (the
 *           GELU/Softmax/RMSNorm crossbar). SRAM-A routing lets the VPU see
 *           either the controller's scratchpad (multi-pass intermediates) or
 *           the requantizer's fresh output for lanes 0..M-1 combined with
 *           externally-supplied ext_sram_b for lanes M..N-1 (so a full
 *           N-wide standalone VPU op like softmax/rmsnorm can run on data
 *           the linear engine didn't produce).
 *
 * @param    M      Parallel output lanes of the linear engine (matmul/dot
 *                    tile height); also sized into the VPU's N-wide crossbar
 *                    for lanes 0..M-1.
 * @param    N      Vector dimension of the nonlinear VPU / contraction tile width.
 * @param    W      Data word width (a_mat/b_vec/ext_sram_b/vpu_data_out).
 * @param    ACC_W  Accumulator width (c_vec bias, dot_acc_out, vpu_dot_out).
 */
module tr_soc_top_int #(
    parameter int M     = 4,   // Parallel Output Lanes
    parameter int N     = 8,   // Vector Dimension
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic clk,
    input  logic rst_n,

    // RISC-V MMIO Interface
    input  logic [7:0]              mmio_addr,
    input  logic [31:0]             mmio_wdata,
    input  logic mmio_wen,
    output logic [31:0]             mmio_rdata,

    // Linear Engine External Inputs
    input  logic dot_in_valid,
    output logic dot_in_ready,
    input  logic [W-1:0]            a_mat [M][N],
    input  logic [W-1:0]            b_vec [N],
    input  logic signed [ACC_W-1:0] c_vec [M],
    // Streaming matmul: on the first tile of a dot drive clear_acc=1 (init the
    // accumulator with c_vec bias); drive 0 on the following tiles to accumulate
    // a[i]*b[i] across an arbitrarily long contraction.  For the legacy
    // single-tile use, tie this high.
    input  logic clear_acc,

    // Non-Linear Engine Memory Interfaces
    input  logic signed [W-1:0]     ext_sram_b [N],
    output logic vpu_out_valid,
    output logic signed [W-1:0]     vpu_data_out [N],

    // Streaming matmul memory interface (CMD=0x04).  The wrapper supplies the
    // weight/activation tiles for (mm_tile_row, mm_tile_col) combinationally and
    // stores mm_out_data[M] for mm_out_row when mm_out_we pulses.
    output logic [15:0]             mm_tile_row,
    output logic [15:0]             mm_tile_col,
    output logic                    mm_mem_rd,
    input  logic                    mm_mem_valid,   // a_tile/b_tile valid this cycle
    input  logic [W-1:0]            mm_a_tile [M][N],
    input  logic [W-1:0]            mm_b_tile [N],
    output logic                    mm_out_we,
    output logic [15:0]             mm_out_row,
    output logic signed [W-1:0]     mm_out_data [M]
);

    // Interconnect Wires
    logic [ACC_W-1:0]                    req_mult;
    logic [5:0]                     req_shift;
    
    // Streaming matmul controller <-> datapath
    logic                           mm_start, mm_done, mm_busy;
    logic [15:0]                    mm_num_rt, mm_num_ct;
    logic                           mm_dot_in_valid, mm_clear_acc;

    // Muxed linear-engine inputs (matmul controller vs legacy single-tile ports)
    logic [W-1:0]                   dot_a_mux [M][N];
    logic [W-1:0]                   dot_b_mux [N];
    logic signed [ACC_W-1:0]        dot_c_mux [M];
    logic                           dot_iv_mux, dot_clr_mux;

    // Dot -> Req Pipe
    logic                           dot_out_valid;
    logic                           req_in_ready;
    logic signed [ACC_W-1:0]        dot_acc_out [M];
    
    // Req -> VPU Pipe
    logic                           req_out_valid;
    logic signed [W-1:0]            req_vec_out [M];

    // VPU Controller -> Datapath Wires
    logic [2:0]                     mux_bb_in_sel;
    logic [1:0]                     mux_mac_a_sel;
    logic [1:0]                     mux_mac_b_sel;
    logic [1:0]                     mux_vecmul_a_sel;
    logic [1:0]                     mux_vecmul_b_sel;
    logic [2:0]                     mux_vpu_out_sel;
    
    logic                           en_piped_max;
    logic                           en_mac_valid;
    logic                           en_vecmul_valid;
    logic                           en_bb_valid;
    
    logic                           mac_clear_acc;
    logic [1:0]                     mac_op_mode;
    logic [1:0]                     vecmul_op_mode;
    logic [1:0]                     vecmul_scale_mode;
    
    logic                           bb_shift_mode;
    logic                           bb_bypass_ln;
    logic                           bb_mode_pre_ln;
    logic [1:0]                     bb_mode_post_ln;
    logic                           sym_mode_en;
    logic signed [W-1:0]            ctrl_scalar_sub_val;
    
    logic                           src_sram_a_sel;
    logic                           src_sram_b_sel;
    logic                           write_ext_sram;

    logic signed [W-1:0]            ctrl_scratch_a [N];
    logic signed [W-1:0]            ctrl_scratch_b [N];

    // VPU Outputs
    logic signed [W-1:0]            vpu_sram_a_in [N];
    logic signed [W-1:0]            vpu_sram_b_in [N];
    logic signed [W-1:0]            vpu_max_out;
    logic signed [ACC_W-1:0]        vpu_dot_out;
    logic signed [W-1:0]            vpu_raw_out [N];
    logic                           vpu_bb_valid;
    logic                           vpu_vecmul_valid;
    logic                           vpu_mac_valid;
    logic                           vpu_max_valid;

    // ---------------------------------------------------------
    // 1. MASTER CONTROLLER
    // ---------------------------------------------------------
    tr_soc_ctrl_int #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) u_ctrl (
        .clk                 (clk),
        .rst_n               (rst_n),
        .mmio_addr           (mmio_addr),
        .mmio_wdata          (mmio_wdata),
        .mmio_wen            (mmio_wen),
        .mmio_rdata          (mmio_rdata),
        
        .req_mult_out        (req_mult),
        .req_shift_out       (req_shift),
        
        .mux_bb_in_sel       (mux_bb_in_sel),
        .mux_mac_a_sel       (mux_mac_a_sel),
        .mux_mac_b_sel       (mux_mac_b_sel),
        .mux_vecmul_a_sel    (mux_vecmul_a_sel),
        .mux_vecmul_b_sel    (mux_vecmul_b_sel),
        .mux_vpu_out_sel     (mux_vpu_out_sel),
        
        .en_piped_max        (en_piped_max),
        .en_mac_valid        (en_mac_valid),
        .en_vecmul_valid     (en_vecmul_valid),
        .en_bb_valid         (en_bb_valid),
        
        .mac_clear_acc       (mac_clear_acc),
        .mac_op_mode         (mac_op_mode),
        .vecmul_op_mode      (vecmul_op_mode),
        .vecmul_scale_mode   (vecmul_scale_mode),
        
        .bb_shift_mode       (bb_shift_mode),
        .bb_bypass_ln        (bb_bypass_ln),
        .bb_mode_pre_ln      (bb_mode_pre_ln),
        .bb_mode_post_ln     (bb_mode_post_ln),
        .sym_mode_en         (sym_mode_en),
        .ctrl_scalar_sub_val (ctrl_scalar_sub_val),
        
        .src_sram_a_sel      (src_sram_a_sel),
        .src_sram_b_sel      (src_sram_b_sel),
        .write_ext_sram      (write_ext_sram),
        .scratch_a_out       (ctrl_scratch_a),
        .scratch_b_out       (ctrl_scratch_b),
        
        .vpu_data_out        (vpu_raw_out),
        .vpu_max_out         (vpu_max_out),
        .vpu_dot_out         (vpu_dot_out),
        
        .vpu_bb_valid        (vpu_bb_valid),
        .vpu_vecmul_valid    (vpu_vecmul_valid),
        .vpu_mac_valid       (vpu_mac_valid),
        .vpu_max_valid       (vpu_max_valid),

        .mm_start            (mm_start),
        .mm_num_row_tiles    (mm_num_rt),
        .mm_num_ctiles       (mm_num_ct),
        .mm_done             (mm_done)
    );

    // ---------------------------------------------------------
    // 1b. STREAMING MATMUL SEQUENCER (shares u_dot + u_req)
    // ---------------------------------------------------------
    tr_matmul_ctrl #(.M(M), .N(N)) u_mm (
        .clk                 (clk),
        .rst_n               (rst_n),
        .start               (mm_start),
        .num_row_tiles       (mm_num_rt),
        .num_ctiles          (mm_num_ct),
        .busy                (mm_busy),
        .done                (mm_done),
        .tile_row            (mm_tile_row),
        .tile_col            (mm_tile_col),
        .mem_rd              (mm_mem_rd),
        .mem_valid           (mm_mem_valid),
        .dot_in_valid        (mm_dot_in_valid),
        .clear_acc           (mm_clear_acc),
        .req_valid           (req_out_valid),
        .result_we           (mm_out_we),
        .result_row          (mm_out_row)
    );
    // Requantizer output is the accumulated dot on the capture cycle.
    assign mm_out_data = req_vec_out;

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

    // ---------------------------------------------------------
    // 2. LINEAR DOT ENGINE
    // ---------------------------------------------------------
    dot_product_engine #(
        .M(M),
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) u_dot (
        .clk                 (clk),
        .rst_n               (rst_n),
        .in_valid            (dot_iv_mux),
        .in_ready            (dot_in_ready),
        .op_mode             (2'b00),
        .a_mat               (dot_a_mux),
        .b_vec               (dot_b_mux),
        .c_vec               (dot_c_mux),
        .clear_acc           (dot_clr_mux),
        .out_valid           (dot_out_valid),
        .out_ready           (req_in_ready),
        .out_vec             (dot_acc_out)
    );

    // ---------------------------------------------------------
    // 3. REQUANTIZER ENGINE
    // ---------------------------------------------------------
    // Note: Parameter M (Lanes) fits into N (Vector Dimension) of the VPU.
    requantize_engine_int #(
        .N(M),
        .ACC_W(ACC_W),
        .MUL_W(32),
        .SHIFT_W(6),
        .OUT_W(W)
    ) u_req (
        .clk                 (clk),
        .rst_n               (rst_n),
        .in_valid            (dot_out_valid),
        .in_ready            (req_in_ready),
        .acc_in              (dot_acc_out),
        .multiplier          (req_mult),
        .shift               (req_shift),
        .out_valid           (req_out_valid),
        .out_ready           (1'b1),
        .out_vec             (req_vec_out)
    );

    // ---------------------------------------------------------
    // 4. NON-LINEAR VPU CROSSBAR
    // ---------------------------------------------------------
    // Top-Level SRAM routing (Controller decides Ext SRAM vs Scratchpad)
    //
    // sram_a lane mapping:
    //   src_sram_a_sel=1  → ctrl_scratch_a[i]  for all N lanes (scratchpad path)
    //   src_sram_a_sel=0  → req_vec_out[i]      for lanes 0..M-1 (requantizer output)
    //                     → ext_sram_b[i]        for lanes M..N-1 (software-supplied
    //                                             via XLR2_SRAMB, allows full N-wide
    //                                             standalone VPU ops: softmax, rmsnorm)
    always_comb begin
        for(int i = 0; i < N; i++) begin
            if (src_sram_a_sel) begin
                vpu_sram_a_in[i] = ctrl_scratch_a[i];
            end else if (i < M) begin
                vpu_sram_a_in[i] = req_vec_out[i];
            end else begin
                vpu_sram_a_in[i] = ext_sram_b[i];
            end

            vpu_sram_b_in[i] = (src_sram_b_sel) ? ctrl_scratch_b[i] : ext_sram_b[i];
        end
    end

    tr_nonlinear_vpu #(
        .N(N),
        .W_VEC(W),
        .W_MAC(16),
        .ACC_W(ACC_W),
        .FRAC_W(4),
        .LUT_IDX_W(3)
    ) u_vpu (
        .clk                 (clk),
        .rst_n               (rst_n),
        .sram_data_a         (vpu_sram_a_in),
        .sram_data_b         (vpu_sram_b_in),
        .vpu_data_out        (vpu_raw_out),
        
        .ctrl_scalar_sub_val (ctrl_scalar_sub_val),
        .vpu_max_out         (vpu_max_out),
        .vpu_dot_out         (vpu_dot_out),
        
        .mux_bb_in_sel       (mux_bb_in_sel),
        .mux_mac_a_sel       (mux_mac_a_sel),
        .mux_mac_b_sel       (mux_mac_b_sel),
        .mux_vecmul_a_sel    (mux_vecmul_a_sel),
        .mux_vecmul_b_sel    (mux_vecmul_b_sel),
        .mux_vpu_out_sel     (mux_vpu_out_sel),
        
        .en_piped_max        (en_piped_max),
        .en_mac_valid        (en_mac_valid),
        .en_vecmul_valid     (en_vecmul_valid),
        .en_bb_valid         (en_bb_valid),
        
        .mac_clear_acc       (mac_clear_acc),
        .mac_op_mode         (mac_op_mode),
        .vecmul_op_mode      (vecmul_op_mode),
        .vecmul_scale_mode   (vecmul_scale_mode),
        
        .bb_shift_mode       (bb_shift_mode),
        .bb_bypass_ln        (bb_bypass_ln),
        .bb_mode_pre_ln      (bb_mode_pre_ln),
        .bb_mode_post_ln     (bb_mode_post_ln),
        .sym_mode_en         (sym_mode_en),
        
        .vpu_bb_valid_out    (vpu_bb_valid),
        .vpu_vecmul_valid_out(vpu_vecmul_valid),
        .vpu_mac_valid_out   (vpu_mac_valid),
        .vpu_max_valid_out   (vpu_max_valid)
    );

    // Final Output Gate
    assign vpu_data_out  = vpu_raw_out;
    assign vpu_out_valid = write_ext_sram;

endmodule