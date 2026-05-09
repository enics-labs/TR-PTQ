`timescale 1ns/1ps

module tr_rmsnorm #(
    parameter int N = 8,
    parameter int W = 8,
    parameter int FRAC_W = 4,
    parameter int ACC_W = 32
)(
    input  logic                 clk,
    input  logic                 rst_n,
    
    input  logic                 valid_in,
    input  logic [1:0]           mode,       
    input  logic signed [W-1:0]  x_in [N],
    input  logic signed [W-1:0]  aux_in [N],   // Used for X (Pass 1), Gamma (Pass 2), InvRMS (Pass 4)
    input  logic signed [ACC_W-1:0] sum_in,    // Used for S (Pass 2)
    input  logic signed [W-1:0]  offset_in,    // Used for LogOffset (Pass 3)

    output logic                 valid_out,
    output logic signed [W-1:0]  y_out [N],
    output logic signed [ACC_W-1:0] sum_out,   // For DOT product
    output logic signed [W-1:0]  ln_out        // For Pass 2 Log output
);

    // ========================================================================
    // DATAPATH STAGE 1: LOG-DOMAIN GENERATION (Combinational)
    // ========================================================================
    logic signed [W-1:0] ln_out_comb [1];
    logic signed [W-1:0] post_ln_out [1];
    
    // sum_in is Qx.8 (from Q4.4 * Q4.4). We shift [19:4] to feed Q12.4 to the ALU.
    tr_ln_alu #(.WIDTH(16), .BITS(FRAC_W), .OUT_WIDTH(W)) u_ln (
        .xq(sum_in[19:4]), .yq(ln_out_comb[0])
    );

    // Mode 10 computes Inverse Square Root (-0.5 * x)
    post_ln_modifier #(.N(1), .W(W)) u_post_ln (
        .x_in(ln_out_comb), .mode_sel(2'b10), .y_out(post_ln_out)
    );

    // ------------------------------------------------------------------------
    // Stage 1 Pipeline Registers
    // ------------------------------------------------------------------------
    logic signed [W-1:0] s1_x [N];
    logic signed [W-1:0] s1_aux [N];
    logic signed [W-1:0] s1_offset;
    logic signed [W-1:0] s1_ln_out;
    logic [1:0]          s1_mode;
    logic                s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_mode <= 2'b00; s1_offset <= '0; s1_ln_out <= '0;
            for(int j=0; j<N; j++) begin s1_x[j] <= '0; s1_aux[j] <= '0; end
        end else begin
            s1_valid  <= valid_in;
            s1_mode   <= mode;
            s1_x      <= x_in;
            s1_aux    <= aux_in;
            s1_offset <= offset_in;
            s1_ln_out <= post_ln_out[0];
        end
    end

    // ========================================================================
    // DATAPATH STAGE 2: EXPONENTIAL ALUS (Combinational)
    // ========================================================================
    logic [2:0]   a_idx [N];
    logic [7:0]   e_a_comb [N];
    logic [7:0]   mantisa_comb [N];
    logic [N-1:0] is_zero_comb;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_exp
            // We only need the exponential for Lane 0 in Pass 3. 
            // We broadcast s1_offset to all to keep layout symmetric.
            tr_exp_alu #(.WIDTH(W), .FRAC_W(FRAC_W), .LUT_IDX_W(3), .ITER(2)) u_exp (
                .x(s1_offset), .a_idx(a_idx[i]), .mantisa(mantisa_comb[i]), .is_zero(is_zero_comb[i])
            );
        end
    endgenerate

    shared_lut_rom #(.N(N)) u_lut (.a_idx(a_idx), .e_a(e_a_comb));

    // ------------------------------------------------------------------------
    // Stage 2 Pipeline Registers
    // ------------------------------------------------------------------------
    logic [7:0]          s2_e_a [N];
    logic [7:0]          s2_mantisa [N];
    logic [N-1:0]        s2_is_zero;
    logic signed [W-1:0] s2_x [N];
    logic signed [W-1:0] s2_aux [N];
    logic signed [W-1:0] s2_ln_out;
    logic [1:0]          s2_mode;
    logic                s2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_mode <= 2'b00; s2_is_zero <= '0; s2_ln_out <= '0;
        end else begin
            s2_valid   <= s1_valid;
            s2_mode    <= s1_mode;
            s2_is_zero <= is_zero_comb;
            s2_e_a     <= e_a_comb;
            s2_mantisa <= mantisa_comb;
            s2_x       <= s1_x;
            s2_aux     <= s1_aux;
            s2_ln_out  <= s1_ln_out;
        end
    end

    // ========================================================================
    // DATAPATH STAGE 3/4: MULTIPLIER & DATA ROUTING
    // ========================================================================
    logic [W-1:0] vec_a_in [N];
    logic [W-1:0] vec_b_in [N];
    logic [1:0]   vec_op_mode;
    logic         vec_elemwise;
    
    always_comb begin
        // Mode 2 (UU) for Pass 3. Mode 0 (SS) for all other passes.
        vec_op_mode = (s2_mode == 2'b10) ? 2'd2 : 2'd0; 
        
        // Mode 00 is DOT product. Everything else is ELEMWISE.
        vec_elemwise = (s2_mode != 2'b00);

        for(int j=0; j<N; j++) begin
            if (s2_mode == 2'b10) begin
                // Pass 3: Reconstruct Exp (e_a * mantisa)
                if (s2_is_zero[j]) begin
                    vec_a_in[j] = 8'd128; 
                    vec_b_in[j] = s2_mantisa[j] << 1; 
                end else begin
                    vec_a_in[j] = s2_e_a[j];
                    vec_b_in[j] = s2_mantisa[j];
                end
            end else begin
                // Pass 1 (X*X), Pass 2 (X*Gamma), Pass 4 (X'*InvRMS)
                vec_a_in[j] = s2_x[j];
                vec_b_in[j] = s2_aux[j];
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
        .mode_elemwise(vec_elemwise),
        .a(vec_a_in), .b(vec_b_in),
        .clear_acc(s2_valid && s2_mode == 2'b00), // Clear DOT accum on valid sum pass
        .out_valid(vec_out_valid), .out_ready(1'b1),
        .out_valid_mask(vec_out_mask), .out_vec(vec_out)
    );

    // ========================================================================
    // STATELESS OUTPUT MUXING
    // ========================================================================
    logic [1:0]          mode_pipe [1:4];
    logic signed [W-1:0] ln_pipe [1:4];

    always_ff @(posedge clk) begin
        mode_pipe[1] <= s2_mode;
        mode_pipe[2] <= mode_pipe[1];
        mode_pipe[3] <= mode_pipe[2];
        mode_pipe[4] <= mode_pipe[3];

        ln_pipe[1] <= s2_ln_out;
        ln_pipe[2] <= ln_pipe[1];
        ln_pipe[3] <= ln_pipe[2];
        ln_pipe[4] <= ln_pipe[3];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0; sum_out <= '0; ln_out <= '0;
            for(int j=0; j<N; j++) y_out[j] <= '0;
        end else begin
            valid_out <= vec_out_valid;
            
            if (vec_out_valid) begin
                if (mode_pipe[4] == 2'b00) begin
                    // Pass 1: Sum of Squares
                    sum_out <= vec_out[0];
                end else if (mode_pipe[4] == 2'b10) begin
                    // Pass 3: Scalar InvRMS (Q0.8 * Q4.4 = Q4.12) -> Slice [15:8]
                    y_out[0] <= vec_out[0][15:8];
                end else begin
                    // Pass 2 & 4: (Q4.4 * Q4.4 = Q8.8) -> Slice [11:4]
                    for(int j=0; j<N; j++) y_out[j] <= vec_out[j][11:4];
                    ln_out <= ln_pipe[4];
                end
            end
        end
    end

endmodule