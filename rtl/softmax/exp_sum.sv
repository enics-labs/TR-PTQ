`define EXP_OP 1'b1
module exp_sum #(
    parameter FRAC = 4,
    parameter ITER = 1,
    parameter int N     = 8,    // vector length
    parameter int W     = 8,   // signed lane width (DSP48-friendly)
    parameter int ACC_W = 32    // accumulator width (DSP48 P width)
) (
    input  logic                    clk,
    input  logic                    rst_n,       // active-low synchronous reset

    // Input stream (vector per beat)
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic                    clear_acc,  // assert with the first beat of a new accumulation window
    input  logic                    op_type,
    input  logic signed [W-1:0]     x1 [N],
    input  logic signed [W-1:0]     x2 [N],

    // Output stream
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic signed [ACC_W-1:0] out_dot
);
    logic [1:0] op_mode;

    logic signed [W-1:0]     a [N];
    logic signed [W-1:0]     b [N];

    logic [W-1:0]     dsp_a_in [N];
    logic [W-1:0]     dsp_b_in [N];

    logic [7:0] e_a     [N];
    logic [7:0] mantisa [N];
    
    assign op_mode = (op_type == `EXP_OP) ? 2'b10 : 2'b00;
    for (genvar i=0; i<N; i++) begin : GEN_EXP
        tr_exp #(.FRAC(FRAC), .ITER(ITER)) u_exp (
            .x(x1[i]),
            .e_a(e_a[i]),
            .mantisa(mantisa[i]),
            .is_zero()
        );
        
        // TODO in the future add operation types if needed like `EXP_OP
        assign a[i] = (op_type == `EXP_OP) ? mantisa[i] : x1[i];
        assign b[i] = (op_type == `EXP_OP) ? e_a[i] : x2[i];

        assign dsp_a_in[i] = a[i];
        assign dsp_b_in[i] = b[i];
    end

// unpacked array [0:7] of packed array [7:0] of logic
// unpacked array [0:7] of signed packed array [7:0] of logic
    vec_mac_dsp48 #(
        .N(N),    // vector length
        .W(W),   // signed lane width (DSP48-friendly)
        .ACC_W(ACC_W)    // accumulator width (DSP48 P width)
    ) dsp_mac(
    .clk(clk),
    .rst_n(rst_n),       // active-low synchronous reset

    // Input stream (vector per beat)
    .in_valid(in_valid),
    .in_ready(in_ready),
    .a(dsp_a_in),
    .b(dsp_b_in),
    .clear_acc(clear_acc),  // assert with the first beat of a new accumulation window
    .op_mode(op_mode),
    // Output stream
    .out_valid(out_valid),
    .out_ready(out_ready),
    .out_dot(out_dot)
    );

endmodule