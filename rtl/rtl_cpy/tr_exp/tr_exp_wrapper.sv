`timescale 1ns/1ps

module tr_exp_wrapper #(
    parameter int N    = 8,
    parameter int FRAC = 4,
    parameter int ITER = 2
)(
    input  logic               dsp_mode, // 1: SoftMax TR-Decomposition, 0: Bypass
    input  logic signed [7:0]  a [N],    // Input vector for exponentiation (and mux A)
    input  logic signed [7:0]  b [N],    // Input vector for mux B

    output logic [7:0]         a_mux [N],
    output logic [7:0]         b_mux [N]
);

    // Internal signals for TR-exp outputs and selection
    logic [7:0] e_a [N];      // Anchor (Q1.7 or Q8)
    logic [7:0] e_frac [N];   // Mantissa (Q1.7 or Q8)
    logic       in_sel [N];   // Selection flag per element

    // ========================================================================
    // STAGE 1: TR-Decomposition Array
    // ========================================================================
    // Instantiates N Taylor-Region exponential calculators and multiplexes
    // their outputs with the standard inputs based on dsp_mode.
    generate
        for (genvar g = 0; g < N; g++) begin : GEN_EXP
            tr_exp #(
                .FRAC (FRAC),
                .ITER (ITER)
            ) u_tr_exp (
                .x       (a[g]),         // signed Q4, <= 0
                .e_a     (e_a[g]),       // Anchor
                .mantisa (e_frac[g]),    // Mantissa
                .is_zero ()              // Unused at this level
            );
            
            // Multiplexing logic
            assign in_sel[g] = (dsp_mode == 1'b1) ? 1'b1 : 1'b0; 
            
            // Route TR-exp outputs if dsp_mode is active, else pass inputs through
            assign a_mux[g] = in_sel[g] ? e_a[g]    : a[g];
            assign b_mux[g] = in_sel[g] ? e_frac[g] : b[g];
        end
    endgenerate

endmodule