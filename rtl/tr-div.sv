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
    output logic [OUT_WIDTH-1:0]           yq    // Q1.7 (from tr_exp)
);

    // ---------------------------------------------------------
    // ln(x) output
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
    // Negate ln(x)
    // ---------------------------------------------------------
    logic signed [WIDTH/2-1:0] neg_ln_x;

    assign neg_ln_x = -ln_x;

    // ---------------------------------------------------------
    // Resize to Q4.4 for tr_exp
    // (Assuming ln output already compatible or truncated)
    // ---------------------------------------------------------
    logic signed [OUT_WIDTH-1:0] neg_ln_x_q4;

    assign neg_ln_x_q4 = neg_ln_x[OUT_WIDTH-1:0];  // explicit truncation

    // ---------------------------------------------------------
    // Exponentiation
    // ---------------------------------------------------------
    tr_exp #(
        .FRAC(BITS),
        .ITER(ITER)
    ) u_exp (
        .clk  (clk),
        .rst_n(rst_n),
        .x    (neg_ln_x_q4),
        .y    (yq)
    );

endmodule
