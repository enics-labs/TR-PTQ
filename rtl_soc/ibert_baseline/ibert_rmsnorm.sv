`timescale 1ns/1ps

/*
 * @module   ibert_rmsnorm
 * @brief    I-BERT-style integer-only RMSNorm, adapting I-BERT's integer
 *           Newton-Raphson square root (Kim et al., "I-BERT: Integer-only
 *           BERT Quantization", ICML 2021 -- Eq. 15, Algorithm 4, used there
 *           for LayerNorm's variance) to RMSNorm's sum-of-squares reduction.
 * @details  This is an ADAPTATION, not a direct lift: I-BERT's own target is
 *           LayerNorm (mean-subtract, variance, then I-SQRT of the variance).
 *           RMSNorm has no mean-subtraction or variance step -- it Newton-
 *           iterates I-SQRT directly on mean(x_j^2). Only Algorithm 4 (I-SQRT)
 *           is I-BERT's; the sum-of-squares reduction and the final x/rms
 *           divide are standard RMSNorm, not covered by the paper.
 *
 *           rms_code (Q4.4) = I_SQRT(sum(x_j^2) >>> 3): since N=8=2^3, and
 *           x_j are already Q4.4 (so x_j^2 is Q8.8), dividing by N first and
 *           taking the integer sqrt directly gives back a Q4.4 value with no
 *           extra rescale multiply needed (mean_sq_code/2^8 = mean(x_real^2),
 *           sqrt(mean_sq_code) = sqrt(mean(x_real^2))*2^4 = rms_code).
 *           y_i = round(x_i * 16 / rms_code), sign handled separately since
 *           the divider is unsigned; rms_code==0 (all-zero input) outputs 0.
 *
 *           I-SQRT's own "floor(n/x_i)" per Newton iteration, and the final
 *           x/rms divide, both use ibert_divider.sv (a conventional restoring
 *           divider) -- a design choice for this baseline, not from I-BERT,
 *           reused sequentially across the sqrt iterations and the N lane
 *           divisions. Only Q4.4 (N=8, W=8, FRAC_W=4) is implemented/verified.
 *
 *           Timing fix: the original sum-of-squares reduction used generic
 *           32-bit signed multiplies and a linear (sequentially-dependent,
 *           depth-N) accumulate, and the I-SQRT initial-guess exponent finder
 *           used a 16-way *linear* priority scan over mean_sq -- both are the
 *           same class of long-combinational-chain bug that made ibert_gelu
 *           miss its 3.3ns reg2reg target. This version narrows the squaring
 *           to its hand-verified 15-bit range, replaces the linear sum with a
 *           balanced 3-level reduction tree (depth log2(N)=3), and replaces
 *           the 16-way linear MSB scan with a 4-level binary-search MSB
 *           finder (hardcoded for MEANSQ_W=16, the only width this module
 *           supports).
 *
 * @param    N       Number of parallel lanes (must be a power of 2 for the
 *                    >>>3 exact-divide-by-N shortcut used here; N=8).
 * @param    W       Data width (Q4.4 native: 8).
 * @param    FRAC_W  Fractional bits (Q4.4 native: 4).
 * @param    DIV_W   ibert_divider operand width.
 */
module ibert_rmsnorm #(
    parameter int N      = 8,
    parameter int W      = 8,
    parameter int FRAC_W = 4,
    parameter int DIV_W  = 24
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [W-1:0] x_in [N],
    output logic valid_out,
    output logic signed [W-1:0] y_out [N]
);

    initial begin
        if (W != 8 || FRAC_W != 4)
            $error("ibert_rmsnorm: only W=8/FRAC_W=4 (Q4.4) is implemented/verified");
        if (N != 8)
            $error("ibert_rmsnorm: the >>>3 divide-by-N shortcut assumes N=8");
    end

    localparam int SUMSQ_W    = 2 * W + $clog2(N);  // sum of N squares of W-bit values
    localparam int MEANSQ_W   = SUMSQ_W - 3;         // >>>3 for N=8
    localparam int MAX_SQRT_ITERS = 16;               // safety cap; paper: converges within 4 for INT32

    typedef enum logic [3:0] {
        IDLE, SUMSQ, SQRT_INIT,
        SQRT_DIV_ISSUE, SQRT_DIV_WAIT, SQRT_STEP, ZERO_OUT,
        LANE_DIV_ISSUE, LANE_DIV_WAIT, FINISH
    } state_t;
    state_t state;

    logic signed [W-1:0]     x_reg [N];
    logic [MEANSQ_W-1:0]     mean_sq;
    logic [MEANSQ_W-1:0]     rms_code;
    logic [MEANSQ_W-1:0]     sqrt_x;      // current Newton-Raphson guess
    logic [$clog2(MAX_SQRT_ITERS+1)-1:0] iter_cnt;
    logic signed [W-1:0]     y_reg [N];
    logic [$clog2(N)-1:0]    lane_cnt;

    // ---------------------------------------------------------------
    // Divider instance, shared: sqrt Newton iterations, then N lane divides
    // ---------------------------------------------------------------
    logic                  div_valid_in;
    logic [DIV_W-1:0]      div_dividend, div_divisor;
    logic                  div_busy, div_valid_out;
    logic [DIV_W-1:0]      div_quotient;

    ibert_divider #(.WIDTH(DIV_W)) u_div (
        .clk(clk), .rst_n(rst_n),
        .valid_in(div_valid_in), .dividend(div_dividend), .divisor(div_divisor),
        .busy(div_busy), .valid_out(div_valid_out), .quotient(div_quotient), .remainder()
    );

    logic [W-1:0] abs_x [N];
    always_comb
        for (int i = 0; i < N; i++)
            abs_x[i] = x_reg[i][W-1] ? W'(-32'(x_reg[i])) : W'(x_reg[i]);

    // NOTE: div_valid_in is a registered pulse set in the *_ISSUE states, so it
    // is actually 1 one cycle later, during the *_WAIT states -- this mux must
    // therefore stay correct across BOTH the ISSUE and WAIT states of each
    // phase (mean_sq/sqrt_x/lane_cnt/rms_code are all stable across that
    // window), not just the ISSUE state, or the divider latches stale operands.
    always_comb begin
        if (state == SQRT_DIV_ISSUE || state == SQRT_DIV_WAIT) begin
            div_dividend = DIV_W'(mean_sq);
            div_divisor  = DIV_W'(sqrt_x);
        end else begin  // LANE_DIV_ISSUE / LANE_DIV_WAIT
            // y_real = x_real/rms_real = (x_code/16)/(rms_code/16) = x_code/rms_code,
            // so y_code = round(y_real*16) = round(x_code*16/rms_code): shift by
            // FRAC_W (4), not by 8 -- rms_code is already a Q4.4 code, not Q0.8.
            div_dividend = DIV_W'((32'(abs_x[lane_cnt]) << FRAC_W) + (32'(rms_code) >> 1));
            div_divisor  = DIV_W'(rms_code);
        end
    end

    // ---------------------------------------------------------------
    // sum(x_j^2): per-lane square hand-sized to its real range (x_reg in
    // [-128,127] -> square in [0,16384], 15 bits), summed via a balanced
    // 3-level reduction tree instead of a linear (depth-N) accumulate.
    // ---------------------------------------------------------------
    logic [14:0] sq [N];
    always_comb
        for (int i = 0; i < N; i++)
            // abs_x is unsigned magnitude (0..128); widen both multiplicands
            // to 9 bits before multiplying so the product isn't truncated to
            // the operands' own (too-narrow) self-determined width.
            sq[i] = 15'(9'(abs_x[i]) * 9'(abs_x[i]));

    logic [15:0] sumsq_l1 [N/2];
    logic [16:0] sumsq_l2 [N/4];
    logic [SUMSQ_W-1:0] sumsq_comb;
    always_comb begin
        for (int i = 0; i < N/2; i++)
            sumsq_l1[i] = 16'(sq[2*i]) + 16'(sq[2*i+1]);
        for (int i = 0; i < N/4; i++)
            sumsq_l2[i] = 17'(sumsq_l1[2*i]) + 17'(sumsq_l1[2*i+1]);
        sumsq_comb = SUMSQ_W'(sumsq_l2[0]) + SUMSQ_W'(sumsq_l2[1]);
    end

    // MSB finder for I-SQRT's x0: 4-level binary search (hardcoded for
    // MEANSQ_W=16) instead of a 16-way linear scan.
    logic [$clog2(MEANSQ_W)-1:0] msb_pos;
    logic [$clog2(MEANSQ_W+1)-1:0] bits_n, ceil_half;
    always_comb begin
        logic [7:0] v8;
        logic [3:0] v4;
        logic [1:0] v2;

        if (mean_sq[15:8] != 8'b0) begin
            msb_pos[3] = 1'b1;
            v8 = mean_sq[15:8];
        end else begin
            msb_pos[3] = 1'b0;
            v8 = mean_sq[7:0];
        end

        if (v8[7:4] != 4'b0) begin
            msb_pos[2] = 1'b1;
            v4 = v8[7:4];
        end else begin
            msb_pos[2] = 1'b0;
            v4 = v8[3:0];
        end

        if (v4[3:2] != 2'b0) begin
            msb_pos[1] = 1'b1;
            v2 = v4[3:2];
        end else begin
            msb_pos[1] = 1'b0;
            v2 = v4[1:0];
        end

        msb_pos[0] = v2[1];

        bits_n    = {1'b0, msb_pos} + 1'b1;
        ceil_half = (bits_n + 1'b1) >> 1;
    end

    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid_out <= 1'b0;
            div_valid_in <= 1'b0;
            mean_sq <= '0; rms_code <= '0; sqrt_x <= '0; iter_cnt <= '0; lane_cnt <= '0;
            for (int i = 0; i < N; i++) begin
                x_reg[i] <= '0; y_reg[i] <= '0; y_out[i] <= '0;
            end
        end else begin
            valid_out    <= 1'b0;
            div_valid_in <= 1'b0;

            case (state)
                IDLE: begin
                    if (valid_in) begin
                        x_reg <= x_in;
                        state <= SUMSQ;
                    end
                end

                SUMSQ: begin
                    mean_sq <= MEANSQ_W'(sumsq_comb >> 3);
                    state   <= SQRT_INIT;
                end

                SQRT_INIT: begin
                    if (mean_sq == '0) begin
                        state <= ZERO_OUT;   // all-zero input -> zero output, skip the sqrt entirely
                    end else begin
                        sqrt_x   <= MEANSQ_W'(1'b1 <<< ceil_half);
                        iter_cnt <= '0;
                        state    <= SQRT_DIV_ISSUE;
                    end
                end

                SQRT_DIV_ISSUE: begin
                    div_valid_in <= 1'b1;
                    state <= SQRT_DIV_WAIT;
                end

                SQRT_DIV_WAIT: begin
                    if (div_valid_out) state <= SQRT_STEP;
                end

                SQRT_STEP: begin
                    // x_next = (x_i + floor(n/x_i)) >>> 1
                    automatic logic [MEANSQ_W:0] x_next;
                    x_next = ({1'b0, sqrt_x} + (MEANSQ_W+1)'(div_quotient)) >> 1;
                    if (x_next >= MEANSQ_W'(sqrt_x) || iter_cnt == MAX_SQRT_ITERS - 1) begin
                        rms_code <= sqrt_x;   // converged (or safety cap): keep the last guess
                        lane_cnt <= '0;
                        state    <= LANE_DIV_ISSUE;
                    end else begin
                        sqrt_x   <= MEANSQ_W'(x_next);
                        iter_cnt <= iter_cnt + 1'b1;
                        state    <= SQRT_DIV_ISSUE;
                    end
                end

                LANE_DIV_ISSUE: begin
                    div_valid_in <= 1'b1;
                    state <= LANE_DIV_WAIT;
                end

                LANE_DIV_WAIT: begin
                    if (div_valid_out) begin
                        automatic logic [DIV_W-1:0] q;
                        automatic logic [W-1:0]     q_clip;
                        q = (div_quotient > DIV_W'((1 <<< (W-1)) - 1)) ? DIV_W'((1 <<< (W-1)) - 1) : div_quotient;
                        q_clip = W'(q);
                        y_reg[lane_cnt] <= x_reg[lane_cnt][W-1] ? W'(-32'(q_clip)) : q_clip;
                        if (lane_cnt == N - 1) begin
                            state <= FINISH;
                        end else begin
                            lane_cnt <= lane_cnt + 1'b1;
                            state    <= LANE_DIV_ISSUE;
                        end
                    end
                end

                ZERO_OUT: begin
                    for (int i = 0; i < N; i++) y_out[i] <= '0;
                    valid_out <= 1'b1;
                    state     <= IDLE;
                end

                FINISH: begin
                    y_out     <= y_reg;
                    valid_out <= 1'b1;
                    state     <= IDLE;
                end
            endcase
        end
    end

endmodule
