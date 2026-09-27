`timescale 1ns/1ps

/*
 * @module   tr_exp_alu
 * @brief    Taylor-Region (TR) exponential ALU: approximates exp(x)
 *           (x expected <= 0) via nearest-LUT-anchor lookup plus a local
 *           Taylor polynomial in the residual delta=x-a.
 * @details  round.sv rounds x to the nearest of shared_lut_rom's anchor
 *           points, returning the LUT index a_idx (for the caller to fetch
 *           e^a from shared_lut_rom), is_zero (x anchors to the most-
 *           negative/underflow point, so exp(x) should be treated as 0),
 *           and is_ceil (whether x rounded up past its anchor, i.e. the
 *           sign of delta). quadratic_divider computes the delta^2/2 term
 *           from x's fractional bits. The Taylor mantissa -- truncated to
 *           order ITER (0: 1.0 only, 1: 1+delta, 2: 1+delta+delta^2/2) --
 *           is assembled from these and returned for the caller to multiply
 *           against shared_lut_rom's e_a, reconstructing exp(x) ~=
 *           e_a * mantisa.
 *
 * @param    WIDTH      I/O word width (x, mantisa).
 * @param    FRAC_W     Fractional bits of x (also the Taylor delta's width).
 * @param    LUT_IDX_W  Width of the shared LUT index (a_idx).
 * @param    ITER       Taylor truncation order: 0=zero-order, 1=linear, 2=quadratic.
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