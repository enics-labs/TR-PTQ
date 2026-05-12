`timescale 1ns/1ps

/*
 * @module   tr_softmax
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 * @param    W               TODO: Add description
 * @param    FRAC_W          TODO: Add description
 * @param    ACC_W           TODO: Add description
 */
module tr_softmax #(
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

    // Auxiliary stateless inputs from Controller
    input  logic signed [W-1:0]  offset_in,  // Used for 'm' (Pass 2) or 'm + ln(S)' (Pass 4)
    input  logic signed [ACC_W-1:0] sum_in,  // Used for 'S' (Pass 3)
    output logic valid_out,
    output logic signed [W-1:0]  y_out [N],
    output logic signed [ACC_W-1:0] sum_out  // Dedicated wide output for DOT product
);

    // ========================================================================
    // MODE 00: PASS 1 - MAX EXTRACTION
    // ========================================================================
    logic                 max_valid;
    logic signed [W-1:0]  max_out;

    piped_max #(
        .NUM_INPUTS(N), .DATA_WIDTH(W)
    ) u_max (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in && (mode == 2'b00)),
        .in_data(x_in),
        .max_out(max_out),
        .valid_out(max_valid)
    );

    // ========================================================================
    // MODE 10: PASS 3 - LOGARITHM 
    // ========================================================================
    logic signed [W-1:0] ln_out_comb;
    logic signed [W-1:0] ln_out_reg;
    logic                ln_valid_reg;

    // sum_in is Qx.12. Shift [23:8] yields Q12.4. 
    // ALU with BITS=4 natively outputs an 8-bit Q4.4 number.
    tr_ln_alu #(
        .WIDTH(16), .BITS(4), .OUT_WIDTH(W)
    ) u_ln (
        .xq(sum_in[23:8]), 
        .yq(ln_out_comb)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ln_out_reg <= '0;
            ln_valid_reg <= 1'b0;
        end else begin
            ln_valid_reg <= valid_in && (mode == 2'b10);
            ln_out_reg   <= ln_out_comb;
        end
    end

    // ========================================================================
    // PASS 2 & 4 DATAPATH: SUBTRACTION -> EXP -> VEC_MUL
    // ========================================================================
    logic signed [W-1:0] x_sub_comb [N];
    
    // Stateless subtraction using the Controller's offset
    scalar_sub #(
        .NUM_INPUTS(N), .DATA_WIDTH(W)
    ) u_sub (
        .in_data(x_in), .sub_val(offset_in), .out_data(x_sub_comb)
    );

    logic signed [W-1:0] s1_val [N];
    logic [1:0]          s1_mode;
    logic                s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_mode <= 2'b00;
            for(int i=0; i<N; i++) s1_val[i] <= '0;
        end else begin
            s1_valid <= valid_in && (mode == 2'b01 || mode == 2'b11);
            s1_mode  <= mode;
            s1_val   <= x_sub_comb;
        end
    end

    logic [2:0] a_idx [N];
    logic [N-1:0] e_a_comb [N];
    logic [N-1:0] mantisa_comb [N];
    logic [N-1:0] is_zero_comb;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_exp
            tr_exp_alu #(
                .WIDTH(W), .FRAC_W(FRAC_W), .LUT_IDX_W(3), .ITER(2)
            ) u_exp (
                .x       (s1_val[i]),
                .a_idx   (a_idx[i]),
                .mantisa (mantisa_comb[i]),
                .is_zero (is_zero_comb[i])
            );
        end
    endgenerate

    shared_lut_rom #(.N(N)) u_lut (.a_idx(a_idx), .e_a(e_a_comb));

    logic [N-1:0]   s2_e_a [N];
    logic [N-1:0]   s2_mantisa [N];
    logic [N-1:0] s2_is_zero;
    logic [1:0]   s2_mode;
    logic         s2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_mode <= 2'b00; s2_is_zero <= '0;
            for(int j=0; j<N; j++) begin s2_e_a[j] <= '0; s2_mantisa[j] <= '0; end
        end else begin
            s2_valid   <= s1_valid;
            s2_mode    <= s1_mode;
            s2_is_zero <= is_zero_comb;
            s2_e_a     <= e_a_comb;
            s2_mantisa <= mantisa_comb;
        end
    end

    // Data Routing for Multiplier
    logic [W-1:0] vec_a_in [N];
    logic [W-1:0] vec_b_in [N];
    
    always_comb begin
        for(int j=0; j<N; j++) begin
            if (s2_is_zero[j]) begin
                // e^0 = 1.0. (128 * 32 = 4096 -> 1.0 in Q4.12)
                vec_a_in[j] = 8'd128; 
                vec_b_in[j] = 8'd32;
            end else begin
                vec_a_in[j] = s2_e_a[j];
                vec_b_in[j] = s2_mantisa[j];
            end
        end
    end

    logic                vec_out_valid;
    logic [N-1:0]        vec_out_mask;
    logic signed [ACC_W-1:0] vec_out [N];

    vec_mul #(
        .N(N), .W(W), .ACC_W(ACC_W)
    ) u_vec_mul (
        .clk            (clk),
        .rst_n          (rst_n),
        .in_valid       (s2_valid), 
        .in_ready       (),
        .op_mode        (2'd2),             // UU Mode 
        .mode_elemwise  (s2_mode == 2'b11), // 0: DOT (Pass 2), 1: ELEMWISE (Pass 4)
        .a              (vec_a_in),
        .b              (vec_b_in),
        .clear_acc      (s2_valid && s2_mode == 2'b01), 
        .out_valid      (vec_out_valid),
        .out_ready      (1'b1),
        .out_valid_mask (vec_out_mask),
        .out_vec        (vec_out)
    );

    // ========================================================================
    // STATELESS OUTPUT MUXING
    // ========================================================================
    always_comb begin
        valid_out = 1'b0;
        sum_out   = '0;
        y_out     = '{default: '0};

        if (mode == 2'b00) begin
            valid_out = max_valid;
            y_out[0]  = max_out; 

        end else if (mode == 2'b10) begin
            valid_out = ln_valid_reg;
            y_out[0]  = ln_out_reg;

        end else if (mode == 2'b01) begin
            valid_out = vec_out_valid && vec_out_mask[0]; 
            sum_out   = vec_out[0];

        end else if (mode == 2'b11) begin
            valid_out = vec_out_valid && vec_out_mask[1]; 
            for (int j = 0; j < N; j++) begin
                // e_a (Q0.8) * mantisa (Q4.4) = Q4.12
                // Shift >> 8 extracts the Q4.4 probability!
                y_out[j] = vec_out[j][15:8];
            end
        end
    end

endmodule