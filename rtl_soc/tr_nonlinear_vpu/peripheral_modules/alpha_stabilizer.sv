/*
 * @module   alpha_stabilizer
 * @brief    Piecewise-linear GELU sigmoid-slope scaler, forced into the
 *           negative domain tr_exp_alu's Taylor-Region backbone supports.
 * @details  Scales |x| by one of four region-dependent coefficients (27,
 *           26, 25, 24 sixteenths, re-scaled to FRAC_W), selected by which
 *           of the |x|<1/2/3 real-valued bands x falls into -- a piecewise
 *           refinement of the constant GELU sigmoid-slope factor (~1.702)
 *           for better accuracy across magnitude ranges. The result is
 *           saturated to the signed W-bit range, then unconditionally
 *           negated to non-positive (out_vec = -|x_scaled|) so it lands in
 *           the negative-only domain shared_lut_rom's anchor table and
 *           tr_exp_alu's Taylor polynomial expect, regardless of the
 *           original sign of x.
 *
 * @param    N       Vector dimension.
 * @param    W       Word width of in_vec/out_vec.
 * @param    FRAC_W  Fractional bits (defines the |x|=1/2/3 region boundaries
 *                     and the coefficient scaling).
 */
module alpha_stabilizer #(
    parameter int N      = 8,
    parameter int W      = 8,
    parameter int FRAC_W = 4
)(
    input  logic signed [W-1:0] in_vec [N],
    output logic signed [W-1:0] out_vec [N]
);

    // Headroom for x*coef; reduces to today's 16 bits at W=8,FRAC_W=4.
    localparam int EXT_W = W + FRAC_W + 4;

    // Coefficients 27,26,25,24 (sixteenths), re-scaled to FRAC_W. Exact
    // (no rounding) since FRAC_W>=4 is assumed, so 2^(FRAC_W-4) is integral.
    localparam int COEF1 = 27 <<< (FRAC_W - 4);
    localparam int COEF2 = 26 <<< (FRAC_W - 4);
    localparam int COEF3 = 25 <<< (FRAC_W - 4);
    localparam int COEF4 = 24 <<< (FRAC_W - 4);

    localparam signed [W-1:0] MAX_OUT = (1 <<< (W-1)) - 1;
    localparam signed [W-1:0] MIN_OUT = -(1 <<< (W-1));

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : GEN_LANES
            logic signed [EXT_W-1:0] x_ext;
            logic signed [EXT_W-1:0] abs_x;
            logic        [EXT_W-1:0] coef_sel;
            logic signed [EXT_W-1:0] x_mult;
            logic signed [W-1:0]     x_scaled;

            always_comb begin
                x_ext = {{(EXT_W-W){in_vec[i][W-1]}}, in_vec[i]};
                abs_x = x_ext[EXT_W-1] ? -x_ext : x_ext;

                // Region boundaries |x| = 1, 2, 3 (real), at this format's LSB
                if (abs_x < (1 <<< FRAC_W))      coef_sel = EXT_W'(COEF1);
                else if (abs_x < (2 <<< FRAC_W)) coef_sel = EXT_W'(COEF2);
                else if (abs_x < (3 <<< FRAC_W)) coef_sel = EXT_W'(COEF3);
                else                             coef_sel = EXT_W'(COEF4);

                x_mult = x_ext * $signed(coef_sel);

                if (x_mult > ($signed({MAX_OUT}) <<< FRAC_W))      x_scaled = MAX_OUT;
                else if (x_mult < ($signed({MIN_OUT}) <<< FRAC_W)) x_scaled = MIN_OUT;
                else                                               x_scaled = W'(x_mult >>> FRAC_W);

                // Stabilization: forced negative absolute value
                out_vec[i] = (x_scaled > 0) ? -x_scaled : x_scaled;
            end
        end
    endgenerate

endmodule
