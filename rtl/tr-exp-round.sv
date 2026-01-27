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
    input  logic signed [7:0]  e_a,      // Q4
    output logic [7:0]         y
);

    // ---------------------------------------------------------
    // Extract integer and fractional parts
    // ---------------------------------------------------------
    logic signed [3:0] a_int;   // integer part (signed)
    logic              idx_zero;   // integer part (signed)
    logic [2:0]        a_idx;   // LUT index (0..7)
    logic [3:0]        x_frac;  // fractional part

    assign x_frac   = x[3:0];
    assign a_int    = x[6:4];     // SIGNED extraction
    assign idx_zero = ~x[7];     // SIGNED extraction
    assign a_idx    = ~a_int;     // 0,-1..-7 -> 0..7

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
    assign exp_lut[7] = 8'd0;   // e^-7

    // ---------------------------------------------------------
    // Stage 0 : LUT read
    // ---------------------------------------------------------
    logic [7:0] e_a;
    assign e_a = (idx_zero) ? 8'd255: exp_lut[a_idx];

    // ---------------------------------------------------------
    // Stage 1 : first-order term
    // ---------------------------------------------------------
    logic [4:0] first_order;
    assign first_order = {1'b1, x_frac};
    // ---------------------------------------------------------
    // Stage 2 : second-order term
    // ---------------------------------------------------------
    logic [2:0] xa_square;
    logic xa_square_0_trm_0; 
    logic xa_square_0_trm_1; 
    logic xa_square_0_trm_2; 
    logic [5:0] second_order;
    
    assign xa_square_0_trm_0 = x_frac[3]&(~x_frac[2])&x_frac[1]; 
    assign xa_square_0_trm_1 = x_frac[3]&x_frac[2]&x_frac[0]; 
    assign xa_square_0_trm_2 = (~x_frac[3])&x_frac[2]&x_frac[1]; 

    assign xa_square[0] = xa_square_0_trm_0 | xa_square_0_trm_1 | xa_square_0_trm_2;
    assign xa_square[1] = x_frac[3]&(~x_frac[2]) | x_frac[3]&x_frac[1];
    assign xa_square[2] = x_frac[3]&x_frac[2];
    assign second_order = first_order + xa_square;

    // ---------------------------------------------------------
    // Stage 3 : final computation
    // ---------------------------------------------------------

    logic [13:0] y1_w1;
    logic [7:0]  y1_w2;

    logic [13:0] y2_w1;
    logic [7:0]  y2_w2;

    assign y1_w2 = y1_w1[12:4]; 
    assign y1_w1 = e_a * first_order;
    
    assign y2_w2 = y2_w1[12:4]; 
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
                    y <= y1_w2; 
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
