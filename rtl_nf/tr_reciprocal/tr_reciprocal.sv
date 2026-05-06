/*
 * @module   tr_reciprocal
 * @brief    Log-Domain Reciprocal & Inverse Square Root Engine
 * @details  Computes generic reciprocal (1/x) or inverse sqrt (1/sqrt(x)) 
 *           using log-domain transformations: y = exp(-ln(x))
 *
 *           Path Pipeline:
 *           xq -> ln(x) -> negate/shift -> exp(-ln(x)) -> Multiply -> yq
 */
`timescale 1ns/1ps

(* preserve *) // Maintain module boundary during Genus Area Reports
module tr_reciprocal #(
    parameter int IN_WIDTH  = 16, // 16 for SoftMax, 20 for LayerNorm
    parameter int OUT_WIDTH = 8,  // 8 for SoftMax, 12 for LayerNorm
    parameter int IN_FRAC   = 4,  // Fractional bits of input
    parameter int OUT_FRAC  = 4,  // Fractional bits of output/mid-stages
    parameter int INV_SQRT  = 0,  // 0: Reciprocal, 1: Inverse Sqrt
    parameter int ITER      = 2   // 0: Zero-Order, 1: Linear, 2: Quadratic
)(
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic [IN_WIDTH-1:0]  xq,
    output logic [OUT_WIDTH-1:0] yq    
);

    // ========================================================================
    // 1. Logarithm Stage: ln(x)
    // ========================================================================
    logic signed [OUT_WIDTH-1:0] ln_x;

    tr_ln #(
        .WIDTH     (IN_WIDTH),
        .BITS      (IN_FRAC),
        .OUT_WIDTH (OUT_WIDTH)
    ) u_ln (
        .xq (xq),
        .yq (ln_x)
    );

    // ========================================================================
    // 2. Math Selector: Negation & Shift
    // ========================================================================
    logic signed [OUT_WIDTH-1:0] neg_ln_x;
    
    always_comb begin
        if (INV_SQRT) begin
            // LayerNorm Mode: y = x^(-0.5). We negate AND shift right by 1.
            neg_ln_x = -(ln_x >>> 1);
        end else begin
            // SoftMax/GELU Mode: y = x^(-1). Just negate.
            neg_ln_x = -ln_x;
        end
    end

    // ========================================================================
    // 3. Exponential Stage: e^(-ln_x)
    // ========================================================================
    logic [OUT_WIDTH-1:0] e_a;
    logic [OUT_WIDTH-1:0] mantisa;
    
    // LUT_IDX_W dynamically sizes based on output width constraints
    localparam int IDX_W = (OUT_WIDTH == 8) ? 3 : 4;
    
    tr_exp #(
        .WIDTH     (OUT_WIDTH),
        .FRAC_W    (OUT_FRAC),
        .LUT_IDX_W (IDX_W),
        .ITER      (ITER)
    ) u_exp (
        .x       (neg_ln_x),
        .e_a     (e_a),
        .mantisa (mantisa),
        .is_zero ()
    );

    // ========================================================================
    // 4. Final Assembly (Combinational)
    // ========================================================================
    logic [(OUT_WIDTH*2)-1:0] mult_result;
    
    assign mult_result = e_a * mantisa;
    assign yq          = mult_result[OUT_WIDTH+OUT_FRAC-1 : OUT_FRAC];

endmodule