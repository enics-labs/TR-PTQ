`timescale 1ns/1ps

module sole_log2exp #(
    parameter int W = 8,
    parameter int FRAC_W = 4
)(
    input  logic signed [W-1:0] x,      // Input (x - x_max), naturally <= 0
    output logic        [W-1:0] exp_out, // Approximated e^x in Q4.4
    output logic        [3:0]   k_out   // Extracted integer shift amount
);

    // 1. Multiply by log2(e) ≈ 1.4375 (1 + 1/2 - 1/16)
    // Using signed arithmetic to maintain negative values
    logic signed [W+1:0] x_scaled;
    assign x_scaled = x + (x >>> 1) - (x >>> 4);
    
    // 2. Extract the integer shift amount (k)
    // Since x is negative, x_scaled is negative. We negate it to get a positive right-shift amount.
    logic [3:0] k;
    assign k = -x_scaled[W-1 : FRAC_W]; 
    assign k_out = k;
    
    // 3. Shift the base value (1.0 in Q4.4 format)
    localparam logic [W-1:0] ONE_Q_FRAC = 1 << FRAC_W;
    
    // Output 2^(-k). If shift is larger than word width, output 0.
    assign exp_out = (k >= W) ? '0 : (ONE_Q_FRAC >> k);

endmodule