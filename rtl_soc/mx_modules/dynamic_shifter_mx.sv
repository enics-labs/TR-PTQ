`timescale 1ns/1ps

/*
 * @module   dynamic_shifter_mx
 * @brief    N-wide bidirectional arithmetic shifter for MX shared-exponent
 *           scaling.
 * @details  Combinationally shifts every lane of data_in by shift_amount
 *           (the MX block's shared exponent), left (shift_dir=1) to expand
 *           a narrow MX mantissa up before the VPU consumes it, or right
 *           (shift_dir=0, sign-preserving) to compress a wide VPU result
 *           back down when the MX formatter re-quantizes it. Each lane is
 *           cast to OUT_W before shifting so left-shifted bits have room to
 *           expand without truncation.
 *
 * @param    N      Number of parallel lanes.
 * @param    IN_W   Input data width (data_in).
 * @param    OUT_W  Output data width (data_out); must be >= IN_W to
 *                   accommodate left-shift expansion without loss.
 */
module dynamic_shifter_mx #(
    parameter int N     = 4,
    parameter int IN_W  = 8,
    parameter int OUT_W = 16 
)(
    input  logic signed [IN_W-1:0]  data_in [N],
    input  logic signed [7:0]       shift_amount, // The Shared Exponent
    input  logic shift_dir,    // 1 = Left (Expand), 0 = Right (Compress)
    output logic signed [OUT_W-1:0] data_out [N]
);

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_shift_lanes
            always_comb begin
                if (shift_dir == 1'b1) begin
                    // LEFT SHIFT (Expansion before VPU)
                    // Cast to OUT_W first so the shifted bits have room to expand safely
                    data_out[i] = OUT_W'(data_in[i]) <<< shift_amount;
                end else begin
                    // RIGHT SHIFT (Compression for MX Formatter Downscale)
                    // Arithmetic shift right preserves the sign bit
                    data_out[i] = OUT_W'(data_in[i]) >>> shift_amount;
                end
            end
        end
    endgenerate

endmodule