// xq
//  │
//  ▼
// tr_ln (ln(x))
//  │
//  ▼
// negate
//  │
//  ▼
// tr_exp (e^(-ln(x)))
//  │
//  ▼
// 1/x

// =============================================================
// TR-RECIPROCAL
// Computes reciprocal using log-domain:
//   y = exp(-ln(x))
//
// xq : fixed-point input (positive)
// yq : fixed-point output
// =============================================================
module tr_reciprocal #(
    parameter WIDTH = 16,
    parameter OUT_WIDTH = 8,
    parameter BITS  = 4,   // fractional bits for ln domain
    parameter ITER = 2
)(
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic [WIDTH-1:0]     xq,
    output logic [OUT_WIDTH-1:0] yq    // The assembled reciprocal
);

    // ---------------------------------------------------------
    // 1. ln(x) output
    // ---------------------------------------------------------
    logic signed [WIDTH/2-1:0] ln_x;

    tr_ln #(
        .WIDTH(WIDTH),
        .BITS (BITS)
    ) u_ln (
        .xq(xq),
        .yq(ln_x)
    );

    // ---------------------------------------------------------
    // 2. Negate ln(x)
    // ---------------------------------------------------------
    logic signed [WIDTH/2-1:0] neg_ln_x;
    assign neg_ln_x = -ln_x;

    logic signed [OUT_WIDTH-1:0] neg_ln_x_q4;
    assign neg_ln_x_q4 = neg_ln_x[OUT_WIDTH-1:0];  // explicit truncation

    // ---------------------------------------------------------
    // 3. Exponentiation
    // ---------------------------------------------------------
    logic [7:0] e_a;
    logic [7:0] mantisa;
    
    tr_exp #(
        .FRAC(BITS),
        .ITER(ITER)
    ) u_exp (
        .x      (neg_ln_x_q4),
        .e_a    (e_a),
        .mantisa(mantisa),
        .is_zero()
    );

    // ---------------------------------------------------------
    // 4. Final Assembly
    // ---------------------------------------------------------
    // Multiply the decoupled parts and shift back to Q-format
    wire [15:0] mult_result = e_a * mantisa;
    assign yq = mult_result >> BITS;

endmodule