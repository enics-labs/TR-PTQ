`timescale 1ns/1ps

/*
 * @module   tr_gelu
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 * @param    W               TODO: Add description
 * @param    FRAC_W          TODO: Add description
 * @param    ACC_W           TODO: Add description
 */
module tr_gelu #(
    parameter int N = 8,
    parameter int W = 8,
    parameter int FRAC_W = 4,
    parameter int ACC_W = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic [1:0]           mode,       
    input  logic signed [W-1:0]  x_in [N],
    input  logic signed [W-1:0]  aux_in [N], 
    output logic valid_out,
    output logic signed [W-1:0]  y_out [N]
);

    // ========================================================================
    // DATAPATH STAGE 1: LOG-DOMAIN GENERATION 
    // ========================================================================
    logic signed [W-1:0] alpha_out [N];
    alpha_stabilizer #(.N(N), .W(W)) u_alpha (
        .in_vec(x_in), .out_vec(alpha_out)
    );

    logic signed [W-1:0] pre_ln_out [N];
    pre_ln_modifier #(.N(N), .WIDTH_IN(W), .FRAC_W(FRAC_W)) u_pre_ln (
        .x_in(x_in), .mode_add_one(1'b1), .y_out(pre_ln_out)
    );

    logic signed [W-1:0] ln_out [N];
    logic signed [W+3:0] ln_xq [N];
    logic signed [W+3:0] ln_yq [N];
    
    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_ln
            // Zero-pad Q4.4 -> Q4.8 to prevent >> truncation loss inside the ALU
            assign ln_xq[i] = {pre_ln_out[i], 4'b0000}; 
            
            tr_ln_alu #(.WIDTH(W+4), .BITS(FRAC_W+4), .OUT_WIDTH(W+4)) u_ln (
                .xq(ln_xq[i]), .yq(ln_yq[i])
            );
            
            // Shift back Q4.8 -> Q4.4 (with rounding factor 12'd8)
            assign ln_out[i] = (ln_yq[i] + 12'd8) >>> 4;
        end
    endgenerate

    logic signed [W-1:0] post_ln_out [N];
    post_ln_modifier #(.N(N), .W(W)) u_post_ln (
        .x_in(ln_out), .mode_sel(2'b01), .y_out(post_ln_out) 
    );

    // ------------------------------------------------------------------------
    // Stage 1 Pipeline Registers
    // ------------------------------------------------------------------------
    logic signed [W-1:0] s1_exp_in [N];
    logic signed [W-1:0] s1_x_raw [N];
    logic signed [W-1:0] s1_aux_in [N];
    logic [1:0]          s1_mode;
    logic                s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_mode <= 2'b00;
            for(int j=0; j<N; j++) begin
                s1_exp_in[j] <= '0; s1_x_raw[j] <= '0; s1_aux_in[j] <= '0;
            end
        end else begin
            s1_valid  <= valid_in;
            s1_mode   <= mode;
            s1_x_raw  <= x_in;
            s1_aux_in <= aux_in;
            for(int j=0; j<N; j++) begin
                if (mode == 2'b00) s1_exp_in[j] <= alpha_out[j];   
                else               s1_exp_in[j] <= post_ln_out[j]; 
            end
        end
    end

    // ========================================================================
    // DATAPATH STAGE 2: EXPONENTIAL & SYMMETRY 
    // ========================================================================
    logic [2:0]   a_idx [N];
    logic [N-1:0]   e_a_comb [N];
    logic [N-1:0]   mantisa_comb [N];
    logic [N-1:0] is_zero_comb;

    generate
        for (i = 0; i < N; i++) begin : gen_exp
            tr_exp_alu #(.WIDTH(W), .FRAC_W(FRAC_W), .LUT_IDX_W(3), .ITER(2)) u_exp (
                .x(s1_exp_in[i]), .a_idx(a_idx[i]), .mantisa(mantisa_comb[i]), .is_zero(is_zero_comb[i])
            );
        end
    endgenerate

    shared_lut_rom #(.N(N)) u_lut (.a_idx(a_idx), .e_a(e_a_comb));

    logic signed [W-1:0] sym_out [N];
    symmetry_modifier #(.N(N), .WIDTH_X(W), .WIDTH_Y(W), .FRAC_W(FRAC_W)) u_sym (
        .x_raw(s1_x_raw), .y_sig(s1_aux_in), .mode_en(1'b1), .sig_corrected(sym_out)
    );

    // ------------------------------------------------------------------------
    // Stage 2 Pipeline Registers
    // ------------------------------------------------------------------------
    logic [N-1:0]          s2_e_a [N];
    logic [N-1:0]          s2_mantisa [N];
    logic [N-1:0]        s2_is_zero;
    logic signed [W-1:0] s2_x_raw [N];
    logic signed [W-1:0] s2_sym_out [N];
    logic [1:0]          s2_mode;
    logic                s2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_mode <= 2'b00; s2_is_zero <= '0;
        end else begin
            s2_valid   <= s1_valid;
            s2_mode    <= s1_mode;
            s2_is_zero <= is_zero_comb;
            s2_e_a     <= e_a_comb;
            s2_mantisa <= mantisa_comb;
            s2_x_raw   <= s1_x_raw;
            s2_sym_out <= sym_out;
        end
    end

    // ========================================================================
    // DATAPATH STAGE 3: RECONSTRUCTION MULTIPLIER
    // ========================================================================
    logic [W-1:0] vec_a_in [N];
    logic [W-1:0] vec_b_in [N];
    logic [1:0]   vec_op_mode;
    
    always_comb begin
        vec_op_mode = (s2_mode == 2'b10) ? 2'd1 : 2'd2; 

        for(int j=0; j<N; j++) begin
            if (s2_mode == 2'b10) begin
                vec_a_in[j] = s2_x_raw[j];
                vec_b_in[j] = s2_sym_out[j];
            end else begin
                // Since output is >> 8, we calculate 128 * (mantisa * 2) = mantisa * 256.
                if (s2_is_zero[j]) begin
                    vec_a_in[j] = 8'd128; 
                    vec_b_in[j] = s2_mantisa[j] << 1; 
                end else begin
                    vec_a_in[j] = s2_e_a[j];
                    vec_b_in[j] = s2_mantisa[j];
                end
            end
        end
    end

    logic                vec_out_valid;
    logic [N-1:0]        vec_out_mask;
    logic signed [ACC_W-1:0] vec_out [N];

    vec_mul #(
        .N(N), .W(W), .ACC_W(ACC_W)
    ) u_vec_mul (
        .clk(clk), .rst_n(rst_n),
        .in_valid(s2_valid), .in_ready(),
        .op_mode(vec_op_mode),
        .mode_elemwise(1'b1),
        .a(vec_a_in), .b(vec_b_in),
        .clear_acc(1'b0), 
        .out_valid(vec_out_valid), .out_ready(1'b1),
        .out_valid_mask(vec_out_mask), .out_vec(vec_out)
    );

    // ========================================================================
    // STATELESS OUTPUT MUXING
    // ========================================================================
    logic [1:0] mode_pipe [1:4];
    always_ff @(posedge clk) begin
        mode_pipe[1] <= s2_mode;
        mode_pipe[2] <= mode_pipe[1];
        mode_pipe[3] <= mode_pipe[2];
        mode_pipe[4] <= mode_pipe[3];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for(int j=0; j<N; j++) y_out[j] <= '0;
        end else begin
            valid_out <= vec_out_valid;
            
            for(int j=0; j<N; j++) begin
                if (mode_pipe[4] == 2'b10) y_out[j] <= vec_out[j][11:4];
                else                       y_out[j] <= vec_out[j][15:8];
            end
        end
    end

endmodule