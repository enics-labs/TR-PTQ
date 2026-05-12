/*
 * @module   pre_ln_modifier
 * @brief    Log-Domain Pre-Conditioner (Vectorized).
 * @details  Primarily used for GELU denominator construction. 
 *           Executes Out = (mode == 1) ? x + 1.0 : x.
 *           Implemented as a zero-cost bit flip (concatenation) at the 
 *           integer boundary.
 *
 * @param    N      Vector dimension.
 * @param    W      Word width.
 * @param    FRAC_W Fractional bits (defines the location of the 1.0 bit).
 */
module pre_ln_modifier #(
    parameter int N         = 8,
    parameter int WIDTH_IN  = 8,
    parameter int FRAC_W    = 4
)(
    input  logic signed [WIDTH_IN-1:0] x_in [N],
    input  logic mode_add_one, // 0: Bypass, 1: Add +1.0
    output logic signed [WIDTH_IN-1:0] y_out [N]
);
    // Hardware 1.0 Constant based on Fractional Width
    localparam logic signed [WIDTH_IN-1:0] ONE_Q = $signed(1 << FRAC_W);

    always_comb begin
        for (int i = 0; i < N; i++) begin
            if (mode_add_one) begin
                y_out[i] = x_in[i] + ONE_Q;
            end else begin
                // Bypass mode (SoftMax Reciprocal, LayerNorm InvSqrt)
                y_out[i] = x_in[i];
            end
        end
    end

endmodule