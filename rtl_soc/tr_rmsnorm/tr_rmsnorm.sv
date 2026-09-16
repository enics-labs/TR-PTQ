`timescale 1ns/1ps

/*
 * @module   tr_rmsnorm
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 * @param    W               TODO: Add description
 * @param    FRAC_W          TODO: Add description
 * @param    ACC_W           TODO: Add description
 */
module tr_rmsnorm #(
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
    input  logic signed [W-1:0]  aux_in [N],   // Used for X (Pass 1), Gamma (Pass 2), InvRMS (Pass 4)
    input  logic signed [ACC_W-1:0] sum_in,    // Used for S (Pass 2)
    input  logic signed [W-1:0]  offset_in,    // Used for LogOffset (Pass 3)
    output logic valid_out,
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
    // SIGN-GUARD (Pass 3 fix): tr_exp_alu / round.sv's "8-bit Negative-Only
    // LUT Indexing" (see round.sv's own comment) was built only for
    // alpha_stabilizer-style inputs, which are always forced non-positive.
    // offset_in (= ctrl_scalar = 0.5*ln(8) - 0.5*ln(Sum x^2), computed by the
    // controller) has no such guarantee -- for small Sum x^2 it goes
    // positive, the negative-only LUT indexing wraps/breaks, and InvRMS
    // collapses toward 0 instead of growing, as it mathematically should.
    // Root cause + fix verified numerically in the bit-true emulation model
    // (tr_math_model.hpp's rmsnorm_hw_model_signguard_fixed(), see
    // docs/iscas_paper_support/); this is that same fix in RTL.
    //
    // Fix: force offset_in non-positive before it ever reaches tr_exp_alu
    // (cheap: comparator + two's-complement negate, mirrors
    // alpha_stabilizer's own "forced negative absolute value" step), and
    // remember that a flip happened. tr_exp_alu then always evaluates the
    // decay-side case it was actually built for -- no change to the shared
    // exp/ln backbone at all. The flip is undone at the OUTPUT stage (see
    // the recip_lut below), not here -- this stage only prepares the sign.
    // ------------------------------------------------------------------------
    logic                 offset_was_positive;
    logic signed [W-1:0]  offset_guarded;

    assign offset_was_positive = ~offset_in[W-1] && (offset_in != '0);
    assign offset_guarded      = offset_was_positive ? -offset_in : offset_in;

    // ------------------------------------------------------------------------
    // Stage 1 Pipeline Registers
    // ------------------------------------------------------------------------
    logic signed [W-1:0] s1_x [N];
    logic signed [W-1:0] s1_aux [N];
    logic signed [W-1:0] s1_offset;
    logic signed [W-1:0] s1_ln_out;
    logic [1:0]          s1_mode;
    logic                s1_valid;
    logic                s1_offset_neg;   // sign-guard fix: was offset_in positive?

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0; s1_mode <= 2'b00; s1_offset <= '0; s1_ln_out <= '0;
            s1_offset_neg <= 1'b0;
            for(int j=0; j<N; j++) begin s1_x[j] <= '0; s1_aux[j] <= '0; end
        end else begin
            s1_valid  <= valid_in;
            s1_mode   <= mode;
            s1_x      <= x_in;
            s1_aux    <= aux_in;
            s1_offset <= offset_guarded;   // sign-guard fix: guarded, not raw offset_in
            s1_ln_out <= post_ln_out[0];
            s1_offset_neg <= offset_was_positive;   // sign-guard fix
        end
    end

    // ========================================================================
    // DATAPATH STAGE 2: EXPONENTIAL ALUS (Combinational)
    // ========================================================================
    logic [2:0]   a_idx [N];
    logic [N-1:0]   e_a_comb [N];
    logic [N-1:0]   mantisa_comb [N];
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
    logic [N-1:0]          s2_e_a [N];
    logic [N-1:0]          s2_mantisa [N];
    logic [N-1:0]        s2_is_zero;
    logic signed [W-1:0] s2_x [N];
    logic signed [W-1:0] s2_aux [N];
    logic signed [W-1:0] s2_ln_out;
    logic [1:0]          s2_mode;
    logic                s2_valid;
    logic                s2_offset_neg;   // sign-guard fix

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0; s2_mode <= 2'b00; s2_is_zero <= '0; s2_ln_out <= '0;
            s2_offset_neg <= 1'b0;
        end else begin
            s2_valid   <= s1_valid;
            s2_mode    <= s1_mode;
            s2_is_zero <= is_zero_comb;
            s2_e_a     <= e_a_comb;
            s2_mantisa <= mantisa_comb;
            s2_x       <= s1_x;
            s2_aux     <= s1_aux;
            s2_ln_out  <= s1_ln_out;
            s2_offset_neg <= s1_offset_neg;   // sign-guard fix
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

    // ------------------------------------------------------------------------
    // SIGN-GUARD (Pass 3 fix), continued: reciprocal LUT. E (the decay-side
    // exp result the guarded path always produces now) realistically ranges
    // 0..16 (Q4.4 "1.0" = 16, is_zero case); sized to 0..31 for headroom.
    // recip_lut[E] = round(256/E), saturated to 127 (max signed 8-bit) --
    // 256 = 16*16, converting Q4.4 E back out through the same Q4.4
    // convention the rest of this module already uses. E=0 guarded to the
    // same saturated max as E=1 (division by zero shouldn't occur given E's
    // real range, but must not corrupt the pipeline if it ever does).
    // New logic scoped entirely to this module -- tr_exp_alu, round,
    // quadratic_divider, shared_lut_rom (the SHARED backbone GELU and
    // Softmax also use) are untouched.
    // ------------------------------------------------------------------------
    function automatic logic [W-1:0] recip_lut(input logic [W-1:0] e);
        case (e)
            8'd0,  8'd1:  recip_lut = 8'd127;
            8'd2:         recip_lut = 8'd127;
            8'd3:         recip_lut = 8'd85;
            8'd4:         recip_lut = 8'd64;
            8'd5:         recip_lut = 8'd51;
            8'd6:         recip_lut = 8'd43;
            8'd7:         recip_lut = 8'd37;
            8'd8:         recip_lut = 8'd32;
            8'd9:         recip_lut = 8'd28;
            8'd10:        recip_lut = 8'd26;
            8'd11:        recip_lut = 8'd23;
            8'd12:        recip_lut = 8'd21;
            8'd13:        recip_lut = 8'd20;
            8'd14:        recip_lut = 8'd18;
            8'd15:        recip_lut = 8'd17;
            8'd16:        recip_lut = 8'd16;
            8'd17:        recip_lut = 8'd15;
            8'd18:        recip_lut = 8'd14;
            8'd19:        recip_lut = 8'd13;
            8'd20:        recip_lut = 8'd13;
            8'd21:        recip_lut = 8'd12;
            8'd22:        recip_lut = 8'd12;
            8'd23:        recip_lut = 8'd11;
            8'd24:        recip_lut = 8'd11;
            8'd25:        recip_lut = 8'd10;
            8'd26:        recip_lut = 8'd10;
            8'd27:        recip_lut = 8'd9;
            8'd28:        recip_lut = 8'd9;
            8'd29:        recip_lut = 8'd9;
            8'd30:        recip_lut = 8'd9;
            8'd31:        recip_lut = 8'd8;
            default:      recip_lut = 8'd127;   // E outside the expected range -- saturate, don't wrap
        endcase
    endfunction

    // ========================================================================
    // STATELESS OUTPUT MUXING
    // ========================================================================
    logic [1:0]          mode_pipe [1:4];
    logic signed [W-1:0] ln_pipe [1:4];
    logic                neg_pipe [1:4];   // sign-guard fix

    always_ff @(posedge clk) begin
        mode_pipe[1] <= s2_mode;
        mode_pipe[2] <= mode_pipe[1];
        mode_pipe[3] <= mode_pipe[2];
        mode_pipe[4] <= mode_pipe[3];

        ln_pipe[1] <= s2_ln_out;
        ln_pipe[2] <= ln_pipe[1];
        ln_pipe[3] <= ln_pipe[2];
        ln_pipe[4] <= ln_pipe[3];

        neg_pipe[1] <= s2_offset_neg;   // sign-guard fix
        neg_pipe[2] <= neg_pipe[1];
        neg_pipe[3] <= neg_pipe[2];
        neg_pipe[4] <= neg_pipe[3];
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
                    // Sign-guard fix: if offset_in (ctrl_scalar) was originally
                    // positive, vec_out[0][15:8] is E = exp(-|ctrl_scalar|) --
                    // the correct decay-side evaluation of the WRONG (negated)
                    // argument. Recover the true (growing) InvRMS via
                    // exp(+t) = 1/exp(-t), i.e. InvRMS = 256/E, via recip_lut.
                    // If it was already non-positive, this is exactly the
                    // original, unmodified behavior.
                    y_out[0] <= neg_pipe[4] ? recip_lut(vec_out[0][15:8])
                                             : vec_out[0][15:8];
                end else begin
                    // Pass 2 & 4: (Q4.4 * Q4.4 = Q8.8) -> Slice [11:4]
                    for(int j=0; j<N; j++) y_out[j] <= vec_out[j][11:4];
                    ln_out <= ln_pipe[4];
                end
            end
        end
    end

endmodule