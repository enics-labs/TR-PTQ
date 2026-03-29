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
// UPDATE
// ===================================================================================
// ORIGINAL BUG (Index Collision):
// Originally, the LUT index was calculated using a 3-bit bitwise inversion of the 
// rounded anchor magnitude: `~rounded_mag[2:0]`. This created an aliasing collision:
//   * Anchor  0 (0000) -> lower 3 bits 000 -> inverted to 111 (Index 7)
//   * Anchor -8 (1000) -> lower 3 bits 000 -> inverted to 111 (Index 7)
// Because index 7 was hardcoded to hold e^0 (255), inputs near -8 (e.g., -7.625) 
// erroneously fetched e^0 instead of e^-8, causing massive approximation errors.
// 
// FIX:
//   1. The `round.sv` module now outputs `is_zero` when the rounded_mag is zero.
//   2. The LUT was shifted to strictly contain negative anchors (-1 to -8).
//      Index 7 now holds the quantized value for e^-8 (8'd0).
//   3. A bypass multiplexer at Stage 0 catches the zero-anchor case and injects 
//      255 (e^0) directly, bypassing the LUT entirely.
// ===================================================================================
module tr_exp #(
    parameter int FRAC = 4,
    parameter int ITER = 2
)(
    input  logic signed [7:0]  x,      // Q4
    output logic [7:0]         e_a,      // Q4
    output logic [7:0]         mantisa,
    output logic               is_zero
);

    // ---------------------------------------------------------
    // Extract integer and fractional parts
    // ---------------------------------------------------------
    logic              is_ceil;   // integer part (signed)
    logic [2:0]        a_idx;   // LUT index (0..7)
    logic [3:0]        x_frac;  // fractional part


    q4_4_round_neg uut (
        .x(x),
        .is_zero(is_zero),
        .is_ceil(is_ceil),
        .fliped_rounded_int(a_idx)
    );

    // ---------------------------------------------------------
    // Exponent LUT : round(e^a * 2^8)
    // ---------------------------------------------------------
    logic [7:0] exp_lut [0:7];

    assign exp_lut[0] = 8'd94;  // e^-1  // not(x) = neg(x) - 1
    assign exp_lut[1] = 8'd35;  // e^-2
    assign exp_lut[2] = 8'd13;  // e^-3
    assign exp_lut[3] = 8'd5;   // e^-4
    assign exp_lut[4] = 8'd2;   // e^-5
    assign exp_lut[5] = 8'd1;   // e^-6
    assign exp_lut[6] = 8'd0;   // e^-7
    assign exp_lut[7] = 8'd0;   // e^-8

    assign x_frac   = x[3:0];

    // ---------------------------------------------------------
    // Stage 0 : LUT read & e^0 bypass
    // ---------------------------------------------------------
    assign e_a = is_zero ? 8'd255 : exp_lut[a_idx];

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
        .x(x_frac),      // 8-bit signed input
        .y(xa_square)       // 2-bit unsigned output: floor(x^2/32)
    );

    assign second_order = {1'b0, first_order} + {4'b0, xa_square};

    // ---------------------------------------------------------
    // Stage 3 : final computation
    // ---------------------------------------------------------

    always_comb begin
        case (ITER)
            0: begin
                mantisa = 8'd0; 
            end
            1: begin
                mantisa = {3'b000, first_order}; 
            end
            2: begin
                mantisa = {2'b00, second_order}; 
            end
            default: 
                mantisa = 'd0; 
        endcase
    end

endmodule