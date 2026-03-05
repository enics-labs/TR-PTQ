module tr_exp_vec #(
    parameter int N    = 8,
    parameter int FRAC = 4,
    parameter int ITER = 1
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  in_valid,
    output logic                  in_ready,

    input  logic signed [7:0]     x   [N],   // Q4.4, x <= 0

    output logic                  out_valid,
    input  logic                  out_ready,

    output logic [7:0]            e_a     [N], // Q0.8 (your LUT scale)
    output logic [7:0]            mantisa [N], // ~Q1.4-ish packed
    output logic [7:0]            exp_q08 [N]  // reconstructed Q0.8 (optional)
);

    // Simple 0-latency ready/valid pass-through (no internal pipeline here).
    // If you later pipeline tr_exp, then add latency registers + queueing.
    assign in_ready  = out_ready || !out_valid;

    // Register the outputs so out_valid is stable under backpressure
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
        end else if (in_ready) begin
            out_valid <= in_valid;
        end
    end

    // Instantiate N lanes
    for (genvar i = 0; i < N; i++) begin : GEN
        logic is_zero_i;

        tr_exp #(.FRAC(FRAC), .ITER(ITER)) u_exp (
            .x      (x[i]),
            .e_a    (e_a[i]),
            .mantisa(mantisa[i]),
            .is_zero(is_zero_i)
        );

        // Reconstruct exp(x) in Q0.8:
        // e_a is scaled by 2^8, mantisa is scaled by 2^FRAC (≈2^4).
        // Product scaled by 2^(8+FRAC). Shift right by FRAC to return to 2^8.
        logic [15:0] prod_u16;
        always_comb begin
            prod_u16  = e_a[i] * mantisa[i];     // UNSIGNED multiply
            exp_q08[i]= (prod_u16 + (1<<(FRAC-1))) >> FRAC; // rounding shift
        end
    end

endmodule
