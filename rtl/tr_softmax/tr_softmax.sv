module tr_softmax #(
    parameter int N = 8,
    parameter int W = 8,
    parameter int ACC_W = 32
)(
    input  logic                 clk,
    input  logic                 rst_n,
    
    // Input Stream (Raw Attention Scores)
    input  logic                 valid_in,
    input  logic signed [W-1:0]  in_data [N],
    
    // Output Stream 1: Raw Exponents (Ready after ~3 cycles)
    output logic                 out_valid_decomp,
    output logic [W-1:0]         out_e_a [N],
    output logic [W-1:0]         out_e_frac [N],

    // Output Stream 2: Reciprocal (Ready after ~8 cycles)
    output logic                 out_valid_sum,
    output logic [7:0]           out_inv_S
);

    // ========================================================================
    // STAGE 1: Max Scale & TR-Decomposition
    // ========================================================================
    // Subtract the max value to prevent overflow, then decompose the 
    // numbers into the Taylor-Region anchor (e_a) and mantissa (e_frac).
    
    logic                   valid_decomp;
    logic signed [W-1:0]    e_a    [N];
    logic signed [W-1:0]    e_frac [N];

    online_sum #(
        .NUM_INPUTS(N),
        .DATA_WIDTH(W),
        .FRAC_W(4),
        .LUT_IDX_W(3),
        .ITER(1)
    ) u_decompose (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .in_data   (in_data),
        .valid_out (valid_decomp),
        .e_a       (e_a),
        .e_frac    (e_frac)
    );

    // ========================================================================
    // STAGE 2: Exponential Summation (The Denominator S)
    // ========================================================================
    // Reuse Vector MAC engine to multiply e_a * e_frac and accumulate 
    // the results across the vector.

    logic                    valid_sum;
    logic signed [ACC_W-1:0] sum_S;

    // Turn e_a and e_frac from signed to unsigned
    logic [W-1:0] mac_in1 [N];
    logic [W-1:0] mac_in2 [N];

    always_comb begin
        for (int i = 0; i < N; i++) begin
            mac_in1[i] = e_a[i];
            mac_in2[i] = e_frac[i];
        end
    end

    vec_mac_su #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) dsp_mac (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(valid_decomp),
        .in_ready(),
        .a(mac_in1),
        .b(mac_in2),
        .clear_acc(valid_decomp), // Clear accumulator on every new vector
        .op_mode(2'b10),        // Mode 2'b10: Unsigned x Unsigned
        .out_valid(valid_sum),
        .out_ready(1'b1),
        .out_dot(sum_S)
    );

    // ========================================================================
    // STAGE 3: Denominator Generator (Reciprocal 1/S)
    // ========================================================================
    // Converts the 32-bit integer sum into the normalization scaling factor.
    // This is purely combinational logic, resolving instantly when sum_S arrives.
    
    logic [7:0] inv_S_raw;
    logic [7:0] inv_S;

    tr_reciprocal #(
        .WIDTH(16),
        .OUT_WIDTH(8),
        .IN_FRAC(4),
        .OUT_FRAC(4),
        .INV_SQRT(0),      // 0 = standard reciprocal 1/x
        .ITER(2)
    ) u_reciprocal (
        .clk   (clk),
        .rst_n (rst_n),
        .xq    (sum_S[23:8]),
        .yq    (inv_S_raw)
    );

    assign inv_S = (sum_S[23:8] <= 16) ? 8'hFF : inv_S_raw;

    // ========================================================================
    // OUTPUT ASSIGNMENTS
    // ========================================================================
    
    // Stream 1 (From exp_x_minus_xmax)
    assign out_valid_decomp = valid_decomp;
    always_comb begin
        for (int i = 0; i < N; i++) begin
            out_e_a[i]    = e_a[i];
            out_e_frac[i] = e_frac[i];
        end
    end

    // Stream 2 (From tr_reciprocal + Overflow Clamp)
    assign out_valid_sum = valid_sum;
    assign out_inv_S = inv_S; 

endmodule