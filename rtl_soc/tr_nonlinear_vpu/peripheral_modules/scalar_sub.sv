`timescale 1ns/1ps

/*
 * @module   scalar_sub
 * @brief    Saturating vector-minus-scalar: out_data[i] = sat(in_data[i] - sub_val).
 * @details  Combinationally subtracts the same scalar (typically the row max
 *           from piped_max, for softmax's max-subtraction step) from every
 *           lane, computing the difference one bit wider than DATA_WIDTH to
 *           safely detect over/underflow before clamping to the signed
 *           DATA_WIDTH range.
 *
 * @param    NUM_INPUTS  Number of parallel input lanes.
 * @param    DATA_WIDTH  Signed data width of in_data/sub_val/out_data.
 */
module scalar_sub #(
    parameter int NUM_INPUTS = 8,
    parameter int DATA_WIDTH = 8
)(
    input  logic signed [DATA_WIDTH-1:0] in_data [NUM_INPUTS],
    input  logic signed [DATA_WIDTH-1:0] sub_val,
    output logic signed [DATA_WIDTH-1:0] out_data [NUM_INPUTS]
);
    // Calculate dynamic saturation bounds based on DATA_WIDTH
    localparam signed [DATA_WIDTH:0] MIN_VAL = -(1 << (DATA_WIDTH-1));
    localparam signed [DATA_WIDTH:0] MAX_VAL =  (1 << (DATA_WIDTH-1)) - 1;

    always_comb begin
        // Declare a 9-bit wire to safely catch mathematical underflow
        logic signed [DATA_WIDTH:0] diff;
        
        for (int j = 0; j < NUM_INPUTS; j++) begin
            diff = $signed({in_data[j][DATA_WIDTH-1], in_data[j]}) - 
                   $signed({sub_val[DATA_WIDTH-1], sub_val});
            
            // Clamp to the most negative value -(2^(W-1)) (e.g., -128 for 8-bit)
            if (diff < MIN_VAL) begin
                out_data[j] = MIN_VAL[DATA_WIDTH-1:0]; // Clamp to floor (e.g., -128)
            end else if (diff > MAX_VAL) begin
                out_data[j] = MAX_VAL[DATA_WIDTH-1:0]; // Clamp to ceiling (e.g., +127)
            end else begin
                out_data[j] = diff[DATA_WIDTH-1:0];    // Safe to pass through
            end
        end
    end

endmodule