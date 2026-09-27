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
 *           2-cycle latency: valid_in latches x_in, the polynomial is
 *           evaluated combinationally, y_out/valid_out register the result.
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

    function automatic signed [63:0] round_shift(input signed [63:0] val, input int shift);
        round_shift = (val + (64'sd1 <<< (shift - 1))) >>> shift;
    endfunction

    // ---------------------------------------------------------------
    // Stage 0: latch input
    // ---------------------------------------------------------------
    logic signed [W-1:0] x_reg [N];
    logic                s0_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_valid <= 1'b0;
            for (int i = 0; i < N; i++) x_reg[i] <= '0;
        end else begin
            s0_valid <= valid_in;
            if (valid_in) x_reg <= x_in;
        end
    end

    // ---------------------------------------------------------------
    // Stage 1: combinational polynomial evaluation, per lane
    // ---------------------------------------------------------------
    logic signed [W-1:0] y_comb [N];

    always_comb begin
        for (int i = 0; i < N; i++) begin
            logic signed [63:0] t_code, q_abs_full, q_abs, delta, delta_sq;
            logic signed [63:0] erf_partial, erf_pos, erf, one_plus_erf, y_raw, y_clip;
            logic qsgn;

            t_code     = round_shift(64'(x_reg[i]) * K1, K1_SHIFT);
            qsgn       = t_code[63];
            q_abs_full = qsgn ? -t_code : t_code;
            q_abs      = (q_abs_full > CLIP_T) ? 64'(CLIP_T) : q_abs_full;

            delta       = q_abs + B_CODE;               // in [B_CODE, 0]
            delta_sq    = delta * delta;                  // in [0, CLIP_T^2]
            erf_partial = round_shift(A_CODE * delta_sq, A_SHIFT);
            erf_pos     = erf_partial + ONE_Q8;            // L(t) for t>=0, Q0.8
            erf         = qsgn ? -erf_pos : erf_pos;       // sgn(t) applied

            one_plus_erf = erf + ONE_Q8;                   // (1+erf(t)) in Q1.8, range [0, 2*ONE_Q8]
            y_raw        = round_shift(64'(x_reg[i]) * one_plus_erf, Y_SHIFT);

            // Saturate to the signed W-bit output range.
            if (y_raw > (64'sd1 <<< (W - 1)) - 1) y_clip = (64'sd1 <<< (W - 1)) - 1;
            else if (y_raw < -(64'sd1 <<< (W - 1))) y_clip = -(64'sd1 <<< (W - 1));
            else y_clip = y_raw;

            y_comb[i] = W'(y_clip);
        end
    end

    // ---------------------------------------------------------------
    // Stage 2: register output
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (int i = 0; i < N; i++) y_out[i] <= '0;
        end else begin
            valid_out <= s0_valid;
            if (s0_valid) y_out <= y_comb;
        end
    end

endmodule
