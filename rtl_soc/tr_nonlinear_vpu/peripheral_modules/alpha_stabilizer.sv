/*
 * @module   alpha_stabilizer
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 * @param    W               TODO: Add description
 * @param    FRAC_W          TODO: Add description
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
