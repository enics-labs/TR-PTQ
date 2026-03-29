module q12_8_quadratic_divider (
    input  wire [7:0] x,      // 8-bit signed input
    output wire [7:0] y       // 2-bit unsigned output: floor(x^2/32)
);

    wire signed [15:0] x_sq;
    assign x_sq = $signed(x) * $signed(x);

    assign y = {1'b0, x_sq[15:9]};

endmodule