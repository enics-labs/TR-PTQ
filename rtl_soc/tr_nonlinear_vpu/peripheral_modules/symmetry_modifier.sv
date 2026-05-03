/**
 * @module   symmetry_modifier
 * @brief    GELU Sigmoid Symmetry Logic (Vectorized).
 * @details  Combinational trick that executes Out = (x_raw < 0) ? (1.0 - y_sig) : y_sig.
 *           Used to restore the full GELU shape from the non-positive TR-EXP domain.
 *           Extracts the sign bit directly from the delayed raw input vector.
 *
 * @param    N         Vector dimension.
 * @param    WIDTH_X   Word width of the raw input x (to check the sign bit).
 * @param    WIDTH_Y   Word width of the sigmoid input/output.
 * @param    FRAC_W    Fractional bit width of y (defines the 1.0 constant).
 */
module symmetry_modifier #(
    parameter int N         = 8,
    parameter int WIDTH_X   = 8,
    parameter int WIDTH_Y   = 8,
    parameter int FRAC_W    = 4
)(
    input  logic signed [WIDTH_X-1:0] x_raw [N],
    input  logic signed [WIDTH_Y-1:0] y_sig [N],
    input  logic                      mode_en, // 1: Apply symmetry, 0: Bypass
    output logic signed [WIDTH_Y-1:0] sig_corrected [N]
);

    // Hardware 1.0 Constant (Cast safely to the exact target width)
    localparam logic signed [WIDTH_Y-1:0] ONE_Q = WIDTH_Y'(1) << FRAC_W;

    always_comb begin
        for (int i = 0; i < N; i++) begin
            if (mode_en && x_raw[i][WIDTH_X-1]) begin
                // If symmetry mode is ON and the raw input was negative (MSB == 1)
                sig_corrected[i] = ONE_Q - y_sig[i];
            end else begin
                // Positive input OR Bypass Mode
                sig_corrected[i] = y_sig[i];
            end
        end
    end

endmodule