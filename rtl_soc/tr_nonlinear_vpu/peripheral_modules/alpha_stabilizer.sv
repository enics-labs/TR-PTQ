/*
 * @module   alpha_stabilizer
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 * @param    W               TODO: Add description
 */
module alpha_stabilizer #(
    parameter int N = 8,
    parameter int W = 8
)(
    input  logic signed [W-1:0] in_vec [N],
    output logic signed [W-1:0] out_vec [N]
);

    localparam int FRAC_W = 4;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : GEN_LANES
            logic [N-1:0]         abs_z;
            logic signed [15:0] x_ext;
            logic signed [15:0] x_base;
            logic signed [15:0] x_mult;
            logic signed [W-1:0] x_scaled;

            always_comb begin
                // 1. Get absolute magnitude to determine region index
                abs_z = (in_vec[i][W-1]) ? -in_vec[i] : in_vec[i];
                
                // Sign-extend input to 16 bits to prevent overflow during shifts
                x_ext = {{8{in_vec[i][W-1]}}, in_vec[i]};

                // 2. Base Multiplier (x * 24) using zero-delay wire shifts
                // 24 = 16 + 8 -> (x << 4) + (x << 3)
                x_base = (x_ext <<< 4) + (x_ext <<< 3);

                // 3. Add the remainder based on the region LUT
                case (abs_z[6:4])
                    3'b000:  x_mult = x_base + (x_ext <<< 1) + x_ext; // * 27 (Base + 2x + 1x)
                    3'b001:  x_mult = x_base + (x_ext <<< 1);         // * 26 (Base + 2x)
                    3'b010:  x_mult = x_base + x_ext;                 // * 25 (Base + 1x)
                    default: x_mult = x_base;                         // * 24 (Base)
                endcase

                // 4. Scale and Saturate to Q4.4 (7.93 to -8.0)
                if (x_mult > 16'sd2032)       x_scaled = 8'sd127;
                else if (x_mult < -16'sd2048) x_scaled = -8'sd128;
                else                          x_scaled = x_mult[11:4];

                // 5. Stabilization: Forced negative absolute value
                out_vec[i] = (x_scaled > 0) ? -x_scaled : x_scaled;
            end
        end
    endgenerate

endmodule