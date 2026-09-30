`timescale 1ns/1ps

/*
 * @module   ibert_gelu
 * @brief    I-BERT-style integer-only GELU (Kim et al., "I-BERT: Integer-only
 *           BERT Quantization", ICML 2021 -- Eq. 8-9, Algorithm 2).
 * @details  i-GELU(x) = x/2 * (1 + L(x/sqrt2)), where L is the clipped
 *           second-order polynomial L(t) = sgn(t)*[a*(clip(|t|,-b)+b)^2 + 1]
 *           approximating erf, with the paper's own coefficients a=-0.2888,
 *           b=-1.769. This is a genuinely standalone, self-contained unit
 *           (no shared exp/ln backbone, no divider) -- purely a squarer, a
 *           couple of integer multiplies and clip/sign comparator logic, per
 *           the paper's own claim that the second-order polynomial needs only
 *           addition and multiplication.
 *
 *           Fixed-point realization: the paper's Algorithm 1/2 track an
 *           arbitrary REAL scale S through each stage (valid for the general
 *           BERT calibration setting, where S is a per-tensor float chosen at
 *           calibration time). For this fixed-point hardware unit, native I/O
 *           is Q4.4 (matching the TR-VPU comparison point throughout this
 *           project), and the internal erf-argument domain (t = x/sqrt2) uses
 *           a fixed Q0.8 fractional precision (FRAC_T=8, chosen the same way
 *           TR-VPU's own ANCHOR_Q_BITS=8 anchors are chosen) instead of the
 *           paper's own symbolic per-stage rescale (Sout = a*S^2, etc.) --
 *           the two are algebraically equivalent, this one is just fully
 *           dyadic (power-of-two scales throughout) so it needs no divider,
 *           which is exactly the property being demonstrated for this
 *           operator. Only Q4.4 (N=8, W=8, FRAC_W=4) is implemented/verified;
 *           other formats are out of scope for this comparison.
 *
 *           5-cycle latency, one multiply-sized chain per stage: S0 latches
 *           x_in; S1 computes t=x/sqrt2 and the clipped erf-argument delta;
 *           S2 squares delta; S3 forms erf(t) and (1+erf(t)); S4 does the
 *           final x*(1+erf(t)) multiply, rescale and output clip. Every
 *           intermediate is sized to its actual (hand-verified) numeric
 *           range, not a generic wide type -- an earlier single-cycle,
 *           unpipelined, 64-bit-everywhere version of this module missed a
 *           3.3ns reg2reg target by ~4.5x (two chained ~50-bit multiplies on
 *           one combinational path); this version fixes that.
 *
 * @param    N       Number of parallel lanes.
 * @param    W       Data width (Q4.4 native: 8).
 * @param    FRAC_W  Fractional bits (Q4.4 native: 4).
 */
module ibert_gelu #(
    parameter int N      = 8,
    parameter int W      = 8,
    parameter int FRAC_W = 4
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
            $error("ibert_gelu: only W=8/FRAC_W=4 (Q4.4) is implemented/verified");
    end

    // ---------------------------------------------------------------
    // Fixed-point constants (Q4.4 I/O, internal Q0.8 erf-argument domain).
    // t_code (Q0.8) = round(x_code * K1 / 2^K1_SHIFT), K1/2^K1_SHIFT ~= 16/sqrt2
    // erf polynomial coefficients a=-0.2888, b=-1.769 quantized to Q0.8.
    // ---------------------------------------------------------------
    localparam int K1        = 2897;  // ~= 16/sqrt(2) * 256
    localparam int K1_SHIFT  = 8;
    localparam int CLIP_T    = 453;   // round(1.769 * 256), also |B_CODE|
    localparam int B_CODE    = -453;  // round(-1.769 * 256)
    localparam int A_CODE    = -74;   // round(-0.2888 * 256)
    localparam int A_SHIFT   = 16;    // A_CODE(Q0.8) * delta_sq(Q0.16) -> Q0.24, >>>16 -> Q0.8
    localparam int ONE_Q8    = 256;
    localparam int Y_SHIFT   = 9;     // x_code(Q4.4) * one_plus_erf(Q1.8) -> /2 (GELU's 1/2) and back to Q4.4

    // Hand-verified numeric ranges (x_code in [-128,127]):
    //   t_code    in [-1449, 1449]          -> 12-bit signed is enough, use 13
    //   delta     in [-453, 0]              -> 10-bit signed
    //   delta_sq  in [0, 453^2=205209]      -> 18-bit unsigned, keep as 19-bit signed (sign=0)
    //   erf_partial (post-shift) in [-232,0] -> 10-bit signed
    //   one_plus_erf in [0, 512]            -> 11-bit signed (sign=0)
    //   y_raw (post-shift) in ~[-128,128]   -> 10-bit signed, then clipped to W bits

    // ---------------------------------------------------------------
    // Stage 0: latch input
    // ---------------------------------------------------------------
    logic signed [W-1:0] x_reg0 [N];
    logic                s0_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_valid <= 1'b0;
            for (int i = 0; i < N; i++) x_reg0[i] <= '0;
        end else begin
            s0_valid <= valid_in;
            if (valid_in) x_reg0 <= x_in;
        end
    end

    // ---------------------------------------------------------------
    // Stage 1: t = x/sqrt2 (Q0.8), clipped erf-argument delta = clip(|t|,CLIP_T)+B_CODE
    // ---------------------------------------------------------------
    logic signed [W-1:0]  x_reg1 [N];
    logic signed [9:0]    delta1 [N];
    logic                 qsgn1  [N];
    logic                 s1_valid;

    always_comb begin
        for (int i = 0; i < N; i++) begin
            logic signed [19:0] t_raw;
            logic signed [19:0] t_shifted;
            logic signed [12:0] t_code;
            logic [10:0]        q_abs_full;
            logic [8:0]         q_abs;

            t_raw     = 20'(x_reg0[i]) * 20'(K1);
            t_shifted = (t_raw + (20'sd1 <<< (K1_SHIFT - 1))) >>> K1_SHIFT;
            t_code    = 13'(t_shifted);

            qsgn1[i]   = t_code[12];
            q_abs_full = qsgn1[i] ? 11'(-t_code) : 11'(t_code);
            q_abs      = (q_abs_full > 11'(CLIP_T)) ? 9'(CLIP_T) : 9'(q_abs_full);

            delta1[i] = {1'b0, q_abs} + 10'(B_CODE);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            for (int i = 0; i < N; i++) begin
                x_reg1[i] <= '0;
            end
        end else begin
            s1_valid <= s0_valid;
            if (s0_valid) x_reg1 <= x_reg0;
        end
    end

    logic signed [9:0] delta_reg [N];
    logic              qsgn_reg  [N];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < N; i++) begin delta_reg[i] <= '0; qsgn_reg[i] <= 1'b0; end
        end else if (s0_valid) begin
            delta_reg <= delta1;
            qsgn_reg  <= qsgn1;
        end
    end

    // ---------------------------------------------------------------
    // Stage 2: delta_sq = delta^2
    // ---------------------------------------------------------------
    logic signed [W-1:0] x_reg2 [N];
    logic signed [18:0]  delta_sq2 [N];
    logic                qsgn2     [N];
    logic                s2_valid;

    always_comb
        for (int i = 0; i < N; i++)
            delta_sq2[i] = 19'(delta_reg[i]) * 19'(delta_reg[i]);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            for (int i = 0; i < N; i++) x_reg2[i] <= '0;
        end else begin
            s2_valid <= s1_valid;
            if (s1_valid) x_reg2 <= x_reg1;
        end
    end

    logic signed [18:0] delta_sq_reg [N];
    logic                qsgn_reg2   [N];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < N; i++) begin delta_sq_reg[i] <= '0; qsgn_reg2[i] <= 1'b0; end
        end else if (s1_valid) begin
            delta_sq_reg <= delta_sq2;
            qsgn_reg2    <= qsgn_reg;
        end
    end

    // ---------------------------------------------------------------
    // Stage 3: erf(t) = sgn(t)*(a*delta^2 + 1), one_plus_erf = 1 + erf(t)
    // ---------------------------------------------------------------
    logic signed [W-1:0] x_reg3 [N];
    logic signed [10:0]  one_plus_erf3 [N];
    logic                s3_valid;

    always_comb begin
        for (int i = 0; i < N; i++) begin
            logic signed [24:0] erf_raw;
            logic signed [24:0] erf_shifted;
            logic signed [9:0]  erf_partial;
            logic signed [10:0] erf_pos, erf;

            erf_raw     = 25'(A_CODE) * 25'(delta_sq_reg[i]);
            erf_shifted = (erf_raw + (25'sd1 <<< (A_SHIFT - 1))) >>> A_SHIFT;
            erf_partial = 10'(erf_shifted);

            erf_pos = 11'(erf_partial) + 11'(ONE_Q8);
            erf     = qsgn_reg2[i] ? -erf_pos : erf_pos;

            one_plus_erf3[i] = erf + 11'(ONE_Q8);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_valid <= 1'b0;
            for (int i = 0; i < N; i++) x_reg3[i] <= '0;
        end else begin
            s3_valid <= s2_valid;
            if (s2_valid) x_reg3 <= x_reg2;
        end
    end

    logic signed [10:0] one_plus_erf_reg [N];
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < N; i++) one_plus_erf_reg[i] <= '0;
        end else if (s2_valid) begin
            one_plus_erf_reg <= one_plus_erf3;
        end
    end

    // ---------------------------------------------------------------
    // Stage 4: y = round(x * (1+erf(t)) / 2^Y_SHIFT), clip to W-bit signed
    // ---------------------------------------------------------------
    logic signed [W-1:0] y_comb [N];

    always_comb begin
        for (int i = 0; i < N; i++) begin
            logic signed [18:0] y_raw_full;
            logic signed [18:0] y_shifted;
            logic signed [9:0]  y_raw;

            y_raw_full = 19'(x_reg3[i]) * 19'(one_plus_erf_reg[i]);
            y_shifted  = (y_raw_full + (19'sd1 <<< (Y_SHIFT - 1))) >>> Y_SHIFT;
            y_raw      = 10'(y_shifted);

            if (y_raw > 10'((1 <<< (W - 1)) - 1)) y_comb[i] = W'((1 <<< (W - 1)) - 1);
            else if (y_raw < -10'(1 <<< (W - 1))) y_comb[i] = -W'(1 <<< (W - 1));
            else y_comb[i] = W'(y_raw);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (int i = 0; i < N; i++) y_out[i] <= '0;
        end else begin
            valid_out <= s3_valid;
            if (s3_valid) y_out <= y_comb;
        end
    end

endmodule
