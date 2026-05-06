/*
 * @module   tr_exp
 * @brief    Integer-only Taylor-Region Exponential Engine
 * @details  Supports up to 2nd-order Taylor expansions using discrete LUT anchors.
 *           Includes zero-detection bypass to cleanly handle e^0 approximations 
 *           and avoid alias collisions on negative anchor boundaries.
 */
`timescale 1ns/1ps

module tr_exp #(
    parameter int WIDTH     = 8,
    parameter int FRAC_W    = 4,
    parameter int LUT_IDX_W = 3,
    parameter int ITER      = 2  // 0: Zero-Order, 1: Linear, 2: Quadratic
)(
    input  wire signed [WIDTH-1:0] x,
    output logic       [WIDTH-1:0] e_a,
    output logic       [WIDTH-1:0] mantisa,
    output logic                   is_zero
);

    // ========================================================================
    // 1. Rounding and Extraction
    // ========================================================================
    wire                 is_ceil;
    wire [LUT_IDX_W-1:0] a_idx;

    round #(
        .WIDTH     (WIDTH),
        .FRAC_W    (FRAC_W),
        .LUT_IDX_W (LUT_IDX_W)
    ) u_round (
        .x         (x),
        .is_zero   (is_zero),
        .is_ceil   (is_ceil),
        .lut_idx   (a_idx)
    );

    // ========================================================================
    // 2. Quadratic Divider (for Taylor expansion)
    // ========================================================================
    wire [FRAC_W-1:0] xa_square;
    
    quadratic_divider #(
        .WIDTH  (WIDTH),
        .FRAC_W (FRAC_W)
    ) u_quad (
        .delta    (x[FRAC_W-1:0]),
        .quad_out (xa_square)
    );

    // ========================================================================
    // 3. Architecture Generation
    // ========================================================================
    generate
        if (WIDTH == 8 && FRAC_W == 4) begin : gen_opt_8bit
            // ---------------------------------------------------------
            // MODE 1: 8-bit SoftMax/GELU Optimization
            // ---------------------------------------------------------
            wire [7:0] exp_lut [0:7];
            assign exp_lut[0] = 8'd94;  // e^-1
            assign exp_lut[1] = 8'd35;  // e^-2
            assign exp_lut[2] = 8'd13;  // e^-3
            assign exp_lut[3] = 8'd5;   // e^-4
            assign exp_lut[4] = 8'd2;   // e^-5
            assign exp_lut[5] = 8'd1;   // e^-6
            assign exp_lut[6] = 8'd0;   // e^-7
            assign exp_lut[7] = 8'd0;   // e^-8

            // Stage 0: LUT read & e^0 bypass
            assign e_a = is_zero ? 8'd255 : exp_lut[a_idx];

            // Stage 1: First-order term (1 + x-a)
            wire [4:0] first_order;
            assign first_order = {~is_ceil, x[3:0]};

            // Stage 2: Second-order term ((x-a)^2 / 2)
            wire [5:0] second_order;
            assign second_order = first_order + xa_square[1:0];

            // Stage 3: Mantissa Formatter
            always_comb begin
                case (ITER)
                    0: mantisa = 8'd0;
                    1: mantisa = {3'b000, first_order};
                    2: mantisa = {2'b00, second_order};
                    default: mantisa = 8'd0;
                endcase
            end

        end else begin : gen_generic_12bit
            // ---------------------------------------------------------
            // MODE 2: 12-bit LayerNorm Generic Structure
            // ---------------------------------------------------------
            logic [11:0] e_a_12b;
            
            // LUT spanning Positive and Negative anchors
            always_comb begin
                case (a_idx)
                    4'sd0:   e_a_12b = 12'd256;  // e^0  = 1.000
                    4'sd1:   e_a_12b = 12'd696;  // e^1  = 2.718
                    4'sd2:   e_a_12b = 12'd1891; // e^2  = 7.389
                    4'sd3:   e_a_12b = 12'd3840; // e^3  = 15.00
                    
                    4'sd15:  e_a_12b = 12'd94;   // e^-1 = 0.367
                    4'sd14:  e_a_12b = 12'd34;   // e^-2 = 0.135
                    4'sd13:  e_a_12b = 12'd12;   // e^-3 = 0.049
                    4'sd12:  e_a_12b = 12'd4;    // e^-4 = 0.018
                    4'sd11:  e_a_12b = 12'd1;    // e^-5 = 0.006
                    default: e_a_12b = 12'd0;    // Flush out of bounds
                endcase
            end
            assign e_a = e_a_12b;

            // Stage 1: First-order term (1+x-a)
            wire [FRAC_W:0] first_order;
            assign first_order = {~is_ceil, x[FRAC_W-1:0]};
            
            // Stage 2: Second-order term ((x-a)^2 / 2)
            wire [FRAC_W+1:0] second_order;
            assign second_order = first_order + xa_square;
            
            // Stage 3: Mantissa Formatter
            always_comb begin
                case (ITER)
                    0: mantisa = 12'd0;
                    1: mantisa = { {(WIDTH - FRAC_W - 1){1'b0}}, first_order };
                    2: mantisa = { {(WIDTH - FRAC_W - 2){1'b0}}, second_order };
                    default: mantisa = 12'd0;
                endcase
            end
        end
    endgenerate

endmodule