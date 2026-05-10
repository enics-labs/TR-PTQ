`timescale 1ns/1ps

module tr_soc_top #(
    parameter int M     = 4,   // Parallel Output Lanes
    parameter int N     = 8,   // Vector Dimension
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // RISC-V MMIO Interface
    input  logic [7:0]              mmio_addr,
    input  logic [31:0]             mmio_wdata,
    input  logic                    mmio_wen,
    output logic [31:0]             mmio_rdata,

    // Linear Engine External Inputs
    input  logic                    dot_in_valid,
    output logic                    dot_in_ready,
    input  logic [W-1:0]            a_mat [M][N],
    input  logic [W-1:0]            b_vec [N],
    input  logic signed [ACC_W-1:0] c_vec [M],
    
    // Non-Linear Engine Memory Interfaces
    input  logic signed [W-1:0]     ext_sram_b [N],
    output logic                    vpu_out_valid,
    output logic signed [W-1:0]     vpu_data_out [N]
);

    // Interconnect Wires
    logic [31:0]                    req_mult;
    logic [5:0]                     req_shift;
    
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

    // ---------------------------------------------------------
    // 1. MASTER CONTROLLER
    // ---------------------------------------------------------
    tr_soc_ctrl #(
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
        .vpu_mac_valid       (vpu_mac_valid)
    );

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
        .in_valid            (dot_in_valid),
        .in_ready            (dot_in_ready),
        .op_mode             (2'b00),
        .a_mat               (a_mat),
        .b_vec               (b_vec),
        .c_vec               (c_vec),
        .clear_acc           (1'b1),
        .out_valid           (dot_out_valid),
        .out_ready           (req_in_ready),
        .out_vec             (dot_acc_out)
    );

    // ---------------------------------------------------------
    // 3. REQUANTIZER ENGINE
    // ---------------------------------------------------------
    // Note: Parameter M (Lanes) fits into N (Vector Dimension) of the VPU.
    requantize_array_engine #(
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
    always_comb begin
        for(int i = 0; i < N; i++) begin
            if (i < M) begin
                vpu_sram_a_in[i] = (src_sram_a_sel) ? ctrl_scratch_a[i] : req_vec_out[i];
            end else begin
                vpu_sram_a_in[i] = '0; 
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
        .vpu_mac_valid_out   (vpu_mac_valid)
    );

    // Final Output Gate
    assign vpu_data_out  = vpu_raw_out;
    assign vpu_out_valid = write_ext_sram;

endmodule