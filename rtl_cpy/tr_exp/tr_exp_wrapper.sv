`timescale 1ns/1ps

module tr_exp_wrapper #(
    parameter int N    = 8,
    parameter int FRAC = 4,
    parameter int ITER = 1
)(
    input  logic signed [7:0] x [N],       // Subtracted input vector
    output logic [7:0]        e_a [N],     // Taylor Anchor output
    output logic [7:0]        mantisa [N]  // Taylor Mantissa output
);

    // ========================================================================
    // Vectorized Taylor-Region Exponential Calculators
    // ========================================================================
    generate
        for (genvar g = 0; g < N; g++) begin : GEN_EXP
            tr_exp #(
                .FRAC (FRAC),
                .ITER (ITER)
            ) u_tr_exp (
                .x       (x[g]),         // signed Q4, <= 0
                .e_a     (e_a[g]),       // Anchor (unsigned)
                .mantisa (mantisa[g]),   // Mantissa (unsigned)
                .is_zero ()              // Unused at this level
            );
        end
    endgenerate

endmodule