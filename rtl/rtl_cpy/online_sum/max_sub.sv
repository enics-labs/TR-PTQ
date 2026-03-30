`timescale 1ns/1ps

module max_sub #(
    parameter int NUM_INPUTS = 8,
    parameter int DATA_WIDTH = 8,
    parameter int CLAMP_MIN  = -(1 << (DATA_WIDTH-1)) // Default: e.g., -128 for 8-bit
)(
    input  logic signed [DATA_WIDTH-1:0] in_data [NUM_INPUTS],
    input  logic signed [DATA_WIDTH-1:0] x_max,
    output logic signed [DATA_WIDTH-1:0] out_data [NUM_INPUTS]
);

    always_comb begin
        logic signed [DATA_WIDTH:0] diff;
        
        for (int j = 0; j < NUM_INPUTS; j++) begin
            diff = $signed({in_data[j][DATA_WIDTH-1], in_data[j]}) - 
                   $signed({x_max[DATA_WIDTH-1], x_max});
            
            if (diff < CLAMP_MIN) begin
                out_data[j] = CLAMP_MIN[DATA_WIDTH-1:0];
            end else begin
                out_data[j] = diff[DATA_WIDTH-1:0];
            end
        end
    end

endmodule