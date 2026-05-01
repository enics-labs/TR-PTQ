module alpha_stabilizer #(
    parameter int N = 8,
    parameter int W = 8
)(
    input  logic signed [W-1:0] in_vec [N],
    output logic signed [W-1:0] out_vec [N]
);

    localparam int FRAC_W = 4;

    generate
        for (genvar i = 0; i < N; i++) begin : GEN_LANES
            logic [7:0] abs_z;
            logic signed [7:0] alpha_g;
            logic signed [15:0] x_mult;
            logic signed [W-1:0] x_scaled;

            always_comb begin
                // 1. Get absolute magnitude to determine region index
                abs_z = (in_vec[i][W-1]) ? -in_vec[i] : in_vec[i];
                
                // 2. Extract integer bits [6:4] for 3-region Alpha LUT
                case (abs_z[6:4])
                    3'b000:  alpha_g = 8'h1B; // ~1.702 in Q4.4
                    3'b001:  alpha_g = 8'h1A;
                    3'b010:  alpha_g = 8'h19;
                    default: alpha_g = 8'h18;
                endcase

                // 3. Scale and Saturate to Q4.4 (7.93 to -8.0)
                // Q4.4 * Q4.4 = Q8.8 result (16 bits)
                x_mult = in_vec[i] * $signed({1'b0, alpha_g});
                
                if ($signed(x_mult) > 16'sd2032)       x_scaled = 8'sd127;
                else if ($signed(x_mult) < -16'sd2048) x_scaled = -8'sd128;
                else                                   x_scaled = x_mult[11:4];

                // 4. Stabilization: Forced negative absolute value
                // This ensures we always compute e^-|x| for the 1-ALU trick
                out_vec[i] = (x_scaled > 0) ? -x_scaled : x_scaled;
            end
        end
    endgenerate

endmodule