// ===================================================================================
// MODULE: vec_mul_array_engine (Element-wise Multiplier)
// ===================================================================================
// Evaluates: Out[i] = a[i] * b[i]
// Used for: SoftMax probability scaling, GELU final gating.
// ===================================================================================
/*
 * @module   vec_mul_array_engine
 * @brief    Pipelined N-wide elementwise multiplier array.
 * @details  Evaluates out_vec[i] = a[i]*b[i] for each of the N lanes over a
 *           3-stage pipeline: (1) register inputs, (2) compute the N
 *           per-lane products (mode-selectable signed/unsigned operands via
 *           op_mode, same convention as mac_array_engine), (3) register the
 *           products, sign-extended/cast to the output width -- used for
 *           softmax probability scaling and GELU's final elementwise gate.
 *
 * @param    N      Number of parallel lanes.
 * @param    W      Input operand width (a/b).
 * @param    ACC_W  Output width (out_vec).
 */
module vec_mul_array_engine #(
    parameter int N     = 16,
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic clk,
    input  logic rst_n,

    // Input stream
    input  logic in_valid,
    output logic in_ready,
    input  logic [1:0]               op_mode,    // 0:SS, 1:SU, 2:UU
    input  logic [W-1:0]             a [N],
    input  logic [W-1:0]             b [N],

    // Output stream
    output logic out_valid,
    input  logic out_ready,
    output logic signed [ACC_W-1:0]  out_vec [N] // Array of scaled outputs
);

    logic advance;
    assign advance = (~out_valid) || (out_valid && out_ready);
    assign in_ready = advance;

    // ============================================================
    // Stage 1: Input Registers
    // ============================================================
    logic [W-1:0] a_reg [N];
    logic [W-1:0] b_reg [N];
    logic [1:0]   mode1_op;
    logic v1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v1       <= 1'b0;
            mode1_op <= 2'd0;
        end else if (advance) begin
            v1       <= in_valid;
            mode1_op <= op_mode;
            for (int i = 0; i < N; i++) begin
                a_reg[i] <= a[i];
                b_reg[i] <= b[i];
            end
        end
    end

    // ============================================================
    // Stage 2: Registered Products
    // ============================================================
    logic signed [(2*W+1):0] prod_reg [N];
    logic v2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v2 <= 1'b0;
        end else if (advance) begin
            v2 <= v1;
            for (int i = 0; i < N; i++) begin
                unique case (mode1_op)
                    2'd0: prod_reg[i] <= $signed(a_reg[i]) * $signed(b_reg[i]);
                    2'd1: prod_reg[i] <= $signed(a_reg[i]) * $signed({1'b0, b_reg[i]});
                    default: prod_reg[i] <= $signed({1'b0, a_reg[i]}) * $signed({1'b0, b_reg[i]});
                endcase
            end
        end
    end

    // ============================================================
    // Stage 3: Unified Output (Cast to target width)
    // ============================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            for (int i = 0; i < N; i++) begin
                out_vec[i] <= '0;
            end
        end else if (advance) begin
            out_valid <= v2;
            if (v2) begin
                for (int i = 0; i < N; i++) begin
                    out_vec[i] <= $signed(prod_reg[i]);
                end
            end
        end
    end

endmodule