`timescale 1ns/1ps

module q12_8_tr_exp #(
    parameter int ITER = 2
)(
    input  wire signed [19:0] x,         // Q12.8 Input
    output wire        [19:0] e_a,       // Q12.8 Anchor Output
    output logic       [19:0] mantisa,   // Q12.8 Mantissa Output
    output wire               is_zero    // Zero flag
);

    // ========================================================================
    // Stage 1: Rounding and Anchor Extraction
    // ========================================================================
    wire       is_ceil;
    wire [3:0] a_idx;

    q12_8_round_neg u_round (
        .x(x),
        .is_zero(is_zero),
        .is_ceil(is_ceil),
        .fliped_rounded_int(a_idx)
    );

    // ========================================================================
    // Stage 2: Quadratic Divider ( (x-a)^2 / 2 )
    // ========================================================================
    wire [7:0] xa_square;
    
    q12_8_quadratic_divider u_quad (
        .x(x[7:0]),
        .y(xa_square)
    );

    // ========================================================================
    // Stage 3: LayerNorm Anchor LUT (Signed Q12.8)
    // ========================================================================
    // In Q12.8 format, the value 1.0 is represented by 2^8 = 256.
    logic [19:0] e_a_lut;
    
    always_comb begin
        case (a_idx)
            4'sd0:  e_a_lut = 20'd256;    // e^0  = 1.000
            4'sd1:  e_a_lut = 20'd696;    // e^1  = 2.718
            4'sd2:  e_a_lut = 20'd1892;   // e^2  = 7.389
            4'sd3:  e_a_lut = 20'd5142;   // e^3  = 20.086
            4'sd4:  e_a_lut = 20'd13977;  // e^4  = 54.598
            4'sd5:  e_a_lut = 20'd37994;  // e^5  = 148.413
            4'sd6:  e_a_lut = 20'd103278; // e^6  = 403.429
            4'sd7:  e_a_lut = 20'd280738; // e^7  = 1096.633
            
            4'sd15: e_a_lut = 20'd94;     // e^-1 = 0.368
            4'sd14: e_a_lut = 20'd35;     // e^-2 = 0.135
            4'sd13: e_a_lut = 20'd13;     // e^-3 = 0.050
            4'sd12: e_a_lut = 20'd5;      // e^-4 = 0.018
            4'sd11: e_a_lut = 20'd2;      // e^-5 = 0.007
            4'sd10: e_a_lut = 20'd1;      // e^-6 = 0.002
            default: e_a_lut = 20'd0;     // Flush to 0
        endcase
    end
    
    // Because index 0 mathematically maps directly to 256 in the LUT, 
    // we do not need the 'is_zero' bypass multiplexer here.
    assign e_a = e_a_lut;

    // ========================================================================
    // Stage 4: Taylor Math Datapath
    // ========================================================================
    // First-order term: 1 + (x - a)
    // By prepending ~is_ceil, we inject the 1.0 or 0.0 integer bit perfectly.
    wire [8:0] first_order;
    assign first_order = {~is_ceil, x[7:0]};

    // Second-order term: (1 + x - a) + (x - a)^2 / 2
    wire [9:0] second_order;
    assign second_order = first_order + xa_square;

    // Mantissa Assignment (Zero-padded back to 20-bit Q12.8 format)
    always_comb begin
        case (ITER)
            0: mantisa = 20'd0;
            1: mantisa = {11'd0, first_order};
            2: mantisa = {10'd0, second_order};
            default: mantisa = 20'd0;
        endcase
    end

endmodule