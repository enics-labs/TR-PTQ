`timescale 1ns/1ps

module max_sub #(
    parameter int NUM_INPUTS = 8,
    parameter int DATA_WIDTH = 8
)(
    input  logic signed [DATA_WIDTH-1:0] in_data [NUM_INPUTS],
    input  logic signed [DATA_WIDTH-1:0] x_max,
    output logic signed [DATA_WIDTH-1:0] out_data [NUM_INPUTS]
);

    always_comb begin
        // Declare a 9-bit wire to safely catch mathematical underflow
        logic signed [DATA_WIDTH:0] diff;
        
        for (int j = 0; j < NUM_INPUTS; j++) begin
            diff = $signed({in_data[j][DATA_WIDTH-1], in_data[j]}) - 
                   $signed({x_max[DATA_WIDTH-1], x_max});
            
            // Clamp to the most negative value -(2^(W-1)) (e.g., -128 for 8-bit)
            if (diff < -(1 << (DATA_WIDTH-1))) begin
                out_data[j] = -(1 << (DATA_WIDTH-1));
            end else begin
                out_data[j] = diff[DATA_WIDTH-1:0];
            end
        end
    end

endmodule