`timescale 1ns/1ps

module dynamic_shifter_mx #(
    parameter int N     = 4,
    parameter int IN_W  = 8,
    parameter int OUT_W = 16 
)(
    input  logic signed [IN_W-1:0]  data_in [N],
    input  logic signed [7:0]       shift_amount, // The Shared Exponent
    input  logic                    shift_dir,    // 1 = Left (Expand), 0 = Right (Compress)
    
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