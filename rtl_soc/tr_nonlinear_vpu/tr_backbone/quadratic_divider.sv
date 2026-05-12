//////////////////////////////////////////////////////////////////
// UPDATE:
//      Two modes created: 
//      MODE 1 - The original optimized 8-bit.
//      MODE 2 - 12-bit generic multiplier for LayerNorm.
//////////////////////////////////////////////////////////////////
/*
 * @module   quadratic_divider
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    WIDTH           TODO: Add description
 * @param    FRAC_W          TODO: Add description
 */
module quadratic_divider #(
    parameter int WIDTH = 8,
    parameter int FRAC_W = 4
)(
    input  logic signed [FRAC_W-1:0] delta, 
    output logic [FRAC_W-1:0] quad_out
);

    generate
        if (WIDTH == 8 && FRAC_W == 4) begin : gen_opt_8bit
            // ----------------------------------------------------------------
            // MODE 1: 8-bit Optimized (Using exact K-map equations)
            // ----------------------------------------------------------------
            wire b3 = delta[3];
            wire b2 = delta[2];
            wire b1 = delta[1];
            wire b0 = delta[0];
            
            wire [1:0] y;
            // y[1] (MSB) Logic: High only when x = -8 (1000 in 4-bit two's complement)
            // Formula: b3 AND NOT b2 AND NOT b1 AND NOT b0
            assign y[1] = b3 & ~b2 & ~b1 & ~b0;

            // y[0] (LSB) Logic: High when x is -7, -6, 6, or 7
            // Formula derived from K-map: (b2 & b1) | (b3 & ~b2 & b0) | (b3 & ~b2 & b1)
            assign y[0] = (b2 & b1) | (b3 & ~b2 & b0) | (b3 & ~b2 & b1);
            
            // Pad the 2-bit result to match the fractional width
            assign quad_out = {2'b00, y};

        end else begin : gen_generic_mult
            // ----------------------------------------------------------------
            // MODE 2: 12-bit Generic (Standard Multiplier for LayerNorm)
            // ----------------------------------------------------------------
            wire signed [(FRAC_W*2)-1:0] delta_sq;
            assign delta_sq = delta * delta;
            
            // Shift right by (FRAC_W + 1) to rescale and divide by 2
            assign quad_out = delta_sq[FRAC_W*2-1 : FRAC_W+1];
        end
    endgenerate

endmodule