module tr_exp_high_order_system #(
    parameter int N = 8,        // Number of parallel lanes
    parameter int W = 8,
    parameter int ACC_W = 32,
    parameter int ITER = 2
)(
    input  logic        clk,
    input  logic        rst_n,
    input  logic        in_valid,
    output logic        in_ready,
    input  logic [7:0]  x_vector [N], // Vector of 8 inputs

    output logic        out_valid,
    input  logic        out_ready,
    output logic [ACC_W-1:0] out_dot  // Sum of e^x_0 + e^x_1 + ...
);

    // Arrays to hold the decomposition results for each lane
    logic [7:0] e_a_vals [N];
    logic [7:0] mantissas [N];
    
    logic signed [W-1:0] mac_a [N];
    logic signed [W-1:0] mac_b [N];

    // ---------------------------------------------------------
    // 1. Generate Block: Create N Decomposers
    // ---------------------------------------------------------
    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_decomposers
            // Using your original tr_exp module to get e_a and mantissa
            tr_exp #(
                .ITER(ITER)
            ) u_decomp (
                .x(x_vector[i]),
                .e_a(e_a_vals[i]),
                .mantisa(mantissas[i]), // Note: keep your original spelling "mantisa"
                .is_zero()             // Unused in this top level
            );

            // 2. Map decomposer outputs to MAC lanes
            // lane_i = e_a * (1 + Taylor_Remainder)
            // We approximate e^x as e_a * mantissa (where mantissa is Q4.4)
            always_comb begin
                mac_a[i] = {{(W-8){1'b0}}, e_a_vals[i]};
                mac_b[i] = {{(W-8){1'b0}}, mantissas[i]};
            end
        end
    endgenerate

    // ---------------------------------------------------------
    // 3. The MAC Engine
    // ---------------------------------------------------------
    vec_mac_dsp48 #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) dsp_engine (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .a(mac_a),
        .b(mac_b),
        .clear_acc(1'b1), // Standard dot product behavior
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_dot(out_dot)
    );

endmodule