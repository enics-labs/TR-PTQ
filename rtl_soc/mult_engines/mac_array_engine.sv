// ===================================================================================
// MODULE: mac_array_engine (Dot-Product MAC with Bias)
// ===================================================================================
// Evaluates: Out = SUM(a[i] * b[i]) + C
// Used for: Matrix Multiplication (Dense Cluster), Variance/SoftMax Sums (VPU)
// ===================================================================================
/*
 * @module   mac_array_engine
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 * @param    W               TODO: Add description
 * @param    ACC_W           TODO: Add description
 */
module mac_array_engine #(
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
    input  logic signed [ACC_W-1:0]  c,          // Scalar Bias (+ C)
    input  logic clear_acc,  // 1: Initialize Acc with C, 0: Accumulate

    // Output stream
    output logic out_valid,
    input  logic out_ready,
    output logic signed [ACC_W-1:0]  out_dot     // Single scalar dot-product output
);

    logic advance;
    assign advance = (~out_valid) || (out_valid && out_ready);
    assign in_ready = advance;

    // ============================================================
    // Stage 1: Input Registers
    // ============================================================
    logic [W-1:0]            a_reg [N];
    logic [W-1:0]            b_reg [N];
    logic signed [ACC_W-1:0] c_reg1;
    logic [1:0]              mode1_op;
    logic v1, clr1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v1       <= 1'b0;
            clr1     <= 1'b0;
            mode1_op <= 2'd0;
            c_reg1   <= '0;
        end else if (advance) begin
            v1       <= in_valid;
            clr1     <= clear_acc;
            mode1_op <= op_mode;
            c_reg1   <= c; // Pipeline the +C bias
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
    logic signed [ACC_W-1:0] c_reg2;
    logic v2, clr2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v2     <= 1'b0;
            clr2   <= 1'b0;
            c_reg2 <= '0;
        end else if (advance) begin
            v2     <= v1;
            clr2   <= clr1;
            c_reg2 <= c_reg1;

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
    // Stage 3: Adder Tree Reduction
    // ============================================================
    logic signed [ACC_W-1:0] vec_sum_comb;
    logic signed [ACC_W-1:0] vec_sum_reg;
    logic signed [ACC_W-1:0] c_reg3;
    logic v3, clr3;

    always_comb begin
        vec_sum_comb = '0;
        for (int i = 0; i < N; i++) begin
            vec_sum_comb += $signed(prod_reg[i]);
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v3          <= 1'b0;
            clr3        <= 1'b0;
            c_reg3      <= '0;
            vec_sum_reg <= '0;
        end else if (advance) begin
            v3          <= v2;
            clr3        <= clr2;
            c_reg3      <= c_reg2;
            vec_sum_reg <= vec_sum_comb;
        end
    end

    // ============================================================
    // Stage 4: Accumulator + Bias (C)
    // ============================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_dot   <= '0;
        end else if (advance) begin
            out_valid <= v3;
            if (v3) begin
                if (clr3)
                    // First cycle: Initialize with the sum PLUS the scalar Bias (C)
                    out_dot <= vec_sum_reg + c_reg3;
                else
                    // Subsequent cycles: Standard accumulation
                    out_dot <= out_dot + vec_sum_reg;
            end
        end
    end

endmodule