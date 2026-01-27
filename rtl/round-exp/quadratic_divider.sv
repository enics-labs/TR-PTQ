module quadratic_divider (
    input  wire [3:0] x,      // 8-bit signed input
    output wire [1:0] y       // 2-bit unsigned output: floor(x^2/32)
);

    // Internal wires for clarity (mapping to bits b3, b2, b1, b0)
    wire b3 = x[3];
    wire b2 = x[2];
    wire b1 = x[1];
    wire b0 = x[0];

    // y[1] (MSB) Logic: High only when x = -8 (1000 in 4-bit two's complement)
    // Formula: b3 AND NOT b2 AND NOT b1 AND NOT b0
    assign y[1] = b3 & ~b2 & ~b1 & ~b0;

    // y[0] (LSB) Logic: High when x is -7, -6, 6, or 7
    // Formula derived from K-map: (b2 & b1) | (b3 & ~b2 & b0) | (b3 & ~b2 & b1)
    assign y[0] = (b2 & b1) | (b3 & ~b2 & b0) | (b3 & ~b2 & b1);

endmodule