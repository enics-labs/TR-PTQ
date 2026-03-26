module tr_softmax #(
    parameter int N = 8,
    parameter int W = 8,
    parameter int ACC_W = 32,
    parameter int MODE = 0,         // 0: Int Max+Sub | 1: Ext Max, Int Sub | 2: Ext Max+Sub
    parameter int RECIP_TYPE = 0    // 0: TR-Reciprocal | 1: Classic Divider | 2: Bypass (MAC Only)
)(
    input  logic                 clk,
    input  logic                 rst_n,
    
    // Input Stream (Raw Attention Scores)
    input  logic                 valid_in,
    input  logic signed [W-1:0]  in_data [N],
    input  logic signed [W-1:0]  ext_x_max, // Driven if MODE == 1
    
    // Output Stream 1: Raw Exponents
    output logic                 out_valid_decomp,
    output logic [W-1:0]         out_e_a [N],
    output logic [W-1:0]         out_e_frac [N],

    // Output Stream 2: Accumulation & Reciprocal
    output logic                    out_valid_sum,
    output logic signed [ACC_W-1:0] out_sum_S, // Exposed for MAC-Only bypass mode
    output logic [7:0]              out_inv_S
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
        .ITER(1),
        .MODE(MODE)
    ) u_decompose (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .in_data   (in_data),
        .ext_x_max (ext_x_max),
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
    logic       final_valid_sum;

    wire [15:0] safe_denominator = (sum_S[23:8] == 0) ? 16'd1 : sum_S[23:8];

    generate
        if (RECIP_TYPE == 0) begin : gen_tr_recip
            // ---------------------------------------------------
            // Option 0: TR-Reciprocal (Combinational Log-Domain)
            // ---------------------------------------------------
            tr_reciprocal #(
                .IN_WIDTH(16),
                .OUT_WIDTH(8),
                .IN_FRAC(4),
                .OUT_FRAC(4),
                .INV_SQRT(0),      // 0 = standard reciprocal 1/x
                .ITER(2)
            ) u_reciprocal (
                .clk   (clk),
                .rst_n (rst_n),
                .xq    (safe_denominator),
                .yq    (inv_S_raw)
            );

        end else if (RECIP_TYPE == 1) begin : gen_classic_recip
            // ---------------------------------------------------
            // Option 1: Classic Divider (Multi-Cycle)
            // ---------------------------------------------------
            logic raw_div_valid;
            logic div_valid_d;

            classic_reciprocal #(
                .IN_WIDTH(16),
                .OUT_WIDTH(8),
                .IN_FRAC(4),
                .OUT_FRAC(4)
            ) u_reciprocal (
                .clk   (clk),
                .rst_n (rst_n),
                .start(valid_sum),
                .xq    (safe_denominator),
                .valid_out(raw_div_valid),
                .yq    (inv_S_raw)
            );

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) div_valid_d <= 1'b0;
                else        div_valid_d <= raw_div_valid;
            end
            
            assign final_valid_sum = raw_div_valid & ~div_valid_d;

        end else begin: gen_bypass_recip
            // ---------------------------------------------------
            // Option 2: Bypass Reciprocal (MAC Sum Only)
            // ---------------------------------------------------
            assign inv_S_raw = 8'd0;
            assign final_valid_sum = valid_sum;
        end
    endgenerate

    // ========================================================================
    // OUTPUT ASSIGNMENTS
    // ========================================================================
    assign out_valid_decomp = valid_decomp;
    assign out_valid_sum    = final_valid_sum;
    assign out_sum_S        = sum_S;
    assign out_inv_S = inv_S_raw;

    always_comb begin
        for (int i = 0; i < N; i++) begin
            out_e_a[i]    = e_a[i];
            out_e_frac[i] = e_frac[i];
        end
    end

endmodule