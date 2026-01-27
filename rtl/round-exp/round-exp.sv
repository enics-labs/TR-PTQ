`timescale 1ns/1ps

// =============================================================
// TR-EXP : Integer-only Taylor-Region Exponential
//
// Input:
//   x    : signed [7:0], Q4, x <= 0
//   iter : 0 -> e^a
//          1 -> e^a * (1 + x - a)
//          2 -> e^a * (1 + x - a + (x - a)^2 / 2)
//
// LUT:
//   round(e^a * 2^8), a in {0, -1, ..., -7}
//   e^0 saturated to 255
//
// Fraction bits: 4
// =============================================================
module tr_exp #(
    parameter int FRAC = 4,
    parameter int ITER = 2
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic signed [7:0]  x,      // Q4
    output logic [7:0]  e_a,      // Q4
    output logic [3:0]        x_frac,
    output logic [7:0]         y
);

    // ---------------------------------------------------------
    // Extract integer and fractional parts
    // ---------------------------------------------------------
    logic              idx_zero;   // integer part (signed)
    logic              is_ceil;   // integer part (signed)
    logic [2:0]        a_idx;   // LUT index (0..7)
    // logic [3:0]        x_frac;  // fractional part


    q4_4_round_neg uut (
        .x(x),
        .is_zero(idx_zero),
        .is_ceil(is_ceil),
        .fliped_rounded_int(a_idx)
    );

    // ---------------------------------------------------------
    // Exponent LUT : round(e^a * 2^8)
    // ---------------------------------------------------------
    logic [7:0] exp_lut [0:7];

    assign exp_lut[0] = 8'd94; // e^0
    assign exp_lut[1] = 8'd35;  // e^-1
    assign exp_lut[2] = 8'd13;  // e^-2
    assign exp_lut[3] = 8'd5;  // e^-3
    assign exp_lut[4] = 8'd2;   // e^-4
    assign exp_lut[5] = 8'd1;   // e^-5
    assign exp_lut[6] = 8'd0;   // e^-6
    assign exp_lut[7] = 8'd255;   // e^-7
    assign x_frac   = x[3:0];

    // ---------------------------------------------------------
    // Stage 0 : LUT read
    // ---------------------------------------------------------
    assign e_a = exp_lut[a_idx];

    // ---------------------------------------------------------
    // Stage 1 : first-order term 1+x-a
    // ---------------------------------------------------------
    logic [4:0] first_order;
    assign first_order = {~is_ceil, x_frac};
    // ---------------------------------------------------------
    // Stage 2 : second-order term (x-a)^2/2
    // ---------------------------------------------------------
    logic [1:0] xa_square;
    logic [5:0] second_order;
    quadratic_divider qd(
        .x(x),      // 8-bit signed input
        .y(xa_square)       // 2-bit unsigned output: floor(x^2/32)
    );
    assign second_order = first_order + xa_square;

    // ---------------------------------------------------------
    // Stage 3 : final computation
    // ---------------------------------------------------------

    logic [12:0] y1_w1;
    logic [7:0]  y1_w2;

    logic [12:0] y2_w1;
    logic [7:0]  y2_w2;

    assign y1_w2 = y1_w1[11:4]; 
    assign y1_w1 = e_a * first_order;
    
    assign y2_w2 = y2_w1[11:4]; 
    assign y2_w1 = e_a * second_order;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            y <= 'd0;  
        else begin
            case (ITER)
                0: begin
                    y <= e_a; 
                end
                1: begin
                    y <= (idx_zero) ? exp_lut[7] : y1_w2; 
                end
                2: begin
                    y <= y2_w2; 
                end
                default: 
                    y <= 'd0; 
            endcase
        end
    end

endmodule

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