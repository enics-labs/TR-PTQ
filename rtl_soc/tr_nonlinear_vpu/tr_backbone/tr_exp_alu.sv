`timescale 1ns/1ps

/*
 * @module   tr_exp_alu
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    WIDTH           TODO: Add description
 * @param    FRAC_W          TODO: Add description
 * @param    LUT_IDX_W       TODO: Add description
 * @param    ITER            TODO: Add description
 */
module tr_exp_alu #(
    parameter int WIDTH = 8,
    parameter int FRAC_W = 4,
    parameter int LUT_IDX_W = 3,
    parameter int ITER = 2          // 0: Zero-Order, 1: Linear, 2: Quadratic
)(
    input  logic signed [WIDTH-1:0] x,
    output logic [LUT_IDX_W-1:0] a_idx,   // Sent out to external Shared ROM
    output logic [WIDTH-1:0] mantisa,
    output logic is_zero
);

    // ---------------------------------------------------------
    // 1. Rounding and Index Generation
    // ---------------------------------------------------------
    wire is_ceil;

    round #(
        .WIDTH(WIDTH),
        .FRAC_W(FRAC_W),
        .LUT_IDX_W(LUT_IDX_W)
    ) u_round (
        .x(x),
        .is_zero(is_zero),
        .is_ceil(is_ceil),
        .lut_idx(a_idx)
    );

    // ---------------------------------------------------------
    // 2. Quadratic Divider
    // ---------------------------------------------------------
    wire [FRAC_W-1:0] xa_square;
    
    quadratic_divider #(
        .WIDTH(WIDTH),
        .FRAC_W(FRAC_W)
    ) u_quad (
        .delta(x[FRAC_W-1:0]),
        .quad_out(xa_square)
    );

    // ---------------------------------------------------------
    // 3. Taylor Polynomial (Mantissa) Datapath
    // ---------------------------------------------------------
    // First-order term (1+x-a)
    wire [FRAC_W:0] first_order;
    assign first_order = {~is_ceil, x[FRAC_W-1:0]};
    
    // Second-order term ((x-a)^2 / 2)
    wire [FRAC_W+1:0] second_order;
    assign second_order = first_order + xa_square;
    
    // ---------------------------------------------------------
    // 4. Mantissa Formatting (Zero-Padded to WIDTH)
    // ---------------------------------------------------------
    always_comb begin
        case (ITER)
            0: mantisa = 1 << FRAC_W;
            1: mantisa = { {(WIDTH - FRAC_W - 1){1'b0}}, first_order };
            2: mantisa = { {(WIDTH - FRAC_W - 2){1'b0}}, second_order };
            default: mantisa = 1 << FRAC_W;
        endcase
    end

endmodule