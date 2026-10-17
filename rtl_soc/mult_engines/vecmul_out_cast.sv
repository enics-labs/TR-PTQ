`ifndef TR_GELU_EN_DEFAULT
`ifdef GELU_ENABLE
`define TR_GELU_EN_DEFAULT 1'b1   // opt-in: GELU support (ViT-style builds). Default: no GELU (GELU and SwiGLU never run together)
`else
`define TR_GELU_EN_DEFAULT 1'b0
`endif
`endif

/*
 * @module   vecmul_out_cast
 * @brief    Output stage of tr_nonlinear_vpu's vec-mul engine: scales / rounds / saturates the N wide products to W bits.
 * @details  Combinational. vecmul_scale_mode selects the cast (these are the encodings the controller drives):
 *             000  low W bits, no scaling
 *             001  >>> FRAC_W, wrapping                       (GELU only)
 *             010  >>> 8                                      (GELU only; anchor Q0.8)
 *             011  softmax probabilities: >>> FRAC_W, unsigned saturate at 255
 *             101  RMSNorm apply: ROUNDED >>> (FRAC_W+4), signed saturate
 *             110  SwiGLU apply: round_shift(acc << 2, sw_shift) with sw_shift >= 0, signed saturate
 *           Split out of tr_nonlinear_vpu so synthesis reports its area on its own.
 *
 * @param    N        Lanes.
 * @param    W        Output width.
 * @param    ACC_W    Width of the engine's result (acc); the VPU passes 20 (the largest result is 20 signed bits).
 * @param    FRAC_W   Fractional bits of the Q-format.
 * @param    GELU_EN  0 (default; macro GELU_ENABLE sets 1): modes 001 / 010 are not built.
 */
module vecmul_out_cast #(
    parameter int N       = 8,
    parameter int W       = 8,
    parameter int ACC_W   = 32,
    parameter int FRAC_W  = 4,
    parameter bit GELU_EN = `TR_GELU_EN_DEFAULT
)(
    input  logic signed [ACC_W-1:0] acc [N],
    input  logic [2:0]              scale_mode,
    input  logic signed [5:0]       sw_shift,
    output logic signed [W-1:0]     out [N]
);

    localparam signed [W-1:0] SAT_MAX = (1 <<< (W-1)) - 1;
    localparam signed [W-1:0] SAT_MIN = -(1 <<< (W-1));

    always_comb begin
        for(int i=0; i<N; i++) begin
            logic signed [ACC_W-1:0] shifted;
            if (scale_mode == 3'b011) begin
                // Softmax Q0.8 UNSIGNED probability output: shift = anchor_q_bits(8) + FRAC_W - 8 = FRAC_W.
                // Saturate at 255 (~0.996) instead of wrapping (a single dominant logit can land exactly on 256).
                shifted = acc[i] >>> FRAC_W;
                out[i] = (shifted > 255) ? W'(255) : W'(shifted);
            end else if (scale_mode == 3'b110) begin
                // SwiGLU apply: prod = acc << 2 (silu was stored with 2 dropped bits), requantized with the per-row
                // gain: rounded right shift by sw_shift (always >= 0: the gain is capped at 10), signed saturate.
                logic signed [ACC_W+3:0] pr;       // acc<<2 (+1 rounding bit): ACC_W+4 is enough
                pr = $signed({{4{acc[i][ACC_W-1]}}, acc[i]}) <<< 2;
                if (sw_shift > 0)       pr = (pr + ((ACC_W+4)'(1) <<< (sw_shift - 1))) >>> sw_shift;
                if (pr > SAT_MAX)       out[i] = SAT_MAX;
                else if (pr < SAT_MIN)  out[i] = SAT_MIN;
                else                    out[i] = W'(pr);
            end else if (scale_mode == 3'b101) begin
                // RMSNorm apply: x * inv_rms (inv_rms has FRAC_W+4 fractional bits) -> Q.FRAC_W, ROUNDED, signed saturate.
                shifted = (acc[i] + (1 <<< (FRAC_W + 3))) >>> (FRAC_W + 4);
                if (shifted > SAT_MAX)       out[i] = SAT_MAX;
                else if (shifted < SAT_MIN)  out[i] = SAT_MIN;
                else                         out[i] = W'(shifted);
            end else if (GELU_EN && scale_mode == 3'b010) begin
                out[i] = W'(acc[i] >>> 8);
            end else if (GELU_EN && scale_mode == 3'b001) begin
                out[i] = W'(acc[i] >>> FRAC_W);
            end else begin
                out[i] = W'(acc[i]);
            end
        end
    end

endmodule
