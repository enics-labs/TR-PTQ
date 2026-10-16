/*
 * @module   post_ln_modifier
 * @brief    Log-Domain Division and InvSqrt Logic (Vectorized).
 * @details  Executes the necessary scalar multiplications in the log domain 
 *           to avoid physical dividers.
 *           Mode 00: Bypass (x)
 *           Mode 01: Division (-1.0 * x)
 *           Mode 10: Inverse Square Root (-0.5 * x), floored halving (used by rtl_baseline)
 *           Mode 11: Inverse Square Root, ROUNDED halving: -((x+1)>>>1) (tr_soc RMSNorm)
 *
 * @param    N      Vector dimension.
 * @param    W      Word width.
 */
module post_ln_modifier #(
    parameter int N = 8,
    parameter int W = 20  // Often wider because it follows TR-ln
)(
    input  logic signed [W-1:0] x_in [N],
    input  logic [1:0]          mode_sel, 
    output logic signed [W-1:0] y_out [N]
);

    always_comb begin
        for (int i = 0; i < N; i++) begin
            case (mode_sel)
                2'b00: begin
                    // Bypass Mode
                    y_out[i] = x_in[i];
                end
                
                2'b01: begin
                    // Division Mode: -1.0 * x
                    // Standard two's complement negation
                    y_out[i] = -x_in[i];
                end
                
                2'b10: begin
                    // Inverse Square Root Mode: -0.5 * x
                    // Arithmetic right shift (preserves sign), then negate
                    // Note: We negate the shifted value to handle 2's complement correctly
                    logic signed [W-1:0] shifted_x;
                    shifted_x = x_in[i] >>> 1;
                    y_out[i]  = -shifted_x;
                end
                
                2'b11: begin
                    // Rounded -0.5 * x: -((x + 1) >>> 1), computed one bit wider so x=MAX does not wrap
                    logic signed [W:0] x_ext;
                    x_ext     = {x_in[i][W-1], x_in[i]} + 1;
                    y_out[i]  = W'(-(x_ext >>> 1));
                end

                default: begin
                    y_out[i] = x_in[i];
                end
            endcase
        end
    end

endmodule