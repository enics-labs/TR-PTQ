// op_mode: 0 (SS) signed*signed; 1 (SU) $signed(a)*$signed({1'b0,b});
// 2 (UU) $signed({1'b0,a})*$signed({1'b0,b}) -- zero-padded to protect the MSB.
// mode_elemwise: 0 = DOT (out_vec[0] = accumulated sum, out_valid_mask bit 0
// only), 1 = ELEMWISE (out_vec[i]=a[i]*b[i] for all i, out_valid_mask all
// ones). Both modes share the same output latency.
/*
 * @module   vec_mul
 * @brief    4-stage pipelined N-wide vector multiplier, dual-mode
 *           (accumulating dot-product or elementwise) with mixed-sign
 *           operand support.
 * @details  See the architecture note above for the op_mode
 *           (SS/SU/UU operand signedness) and mode_elemwise (DOT vs
 *           ELEMWISE output shape) encodings; both modes share the same
 *           4-stage latency (register inputs -> per-lane products ->
 *           adder-tree reduction + pass-through -> output mux). In DOT
 *           mode, clear_acc selects between initializing out_vec[0] with
 *           this beat's reduced sum or accumulating onto it (streaming
 *           K-dimension reduction, as in tr_gelu's reconstruction multiply).
 *
 * @param    N      Number of parallel lanes.
 * @param    W      Input operand width (a/b).
 * @param    ACC_W  Accumulator/output width (out_vec).
 */
module vec_mul #(
    parameter int N     = 16,
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic clk,
    input  logic rst_n,

    // Input stream
    input  logic in_valid,
    output logic in_ready,
    input  logic [1:0]               op_mode,        // 0:SS, 1:SU, 2:UU
    input  logic mode_elemwise,  // 0:DOT, 1:ELEMWISE
    input  logic [W-1:0]             a [N],          // raw bits
    input  logic [W-1:0]             b [N],          // raw bits
    input  logic clear_acc,

    // Output stream
    output logic out_valid,
    input  logic out_ready,
    output logic [N-1:0]             out_valid_mask,
    output logic signed [ACC_W-1:0]  out_vec [N]
);

    // ============================================================
    // Handshake / pipeline control
    // ============================================================
    logic advance;
    assign advance  = (~out_valid) || (out_valid && out_ready);
    assign in_ready = advance;

    // ============================================================
    // Stage 1: input registers
    // ============================================================
    logic [W-1:0] a_reg [N];
    logic [W-1:0] b_reg [N];
    logic [1:0]   mode1_op;
    logic         mode1_elemwise;
    logic         v1;
    logic         clr1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v1             <= 1'b0;
            clr1           <= 1'b0;
            mode1_op       <= 2'd0;
            mode1_elemwise <= 1'b0;
        end else if (advance) begin
            v1             <= in_valid;
            clr1           <= clear_acc;
            mode1_op       <= op_mode;
            mode1_elemwise <= mode_elemwise;
            for (int i = 0; i < N; i++) begin
                a_reg[i] <= a[i];
                b_reg[i] <= b[i];
            end
        end
    end

    // ============================================================
    // Stage 2: registered products (mixed signedness)
    // ============================================================
    // Product width = 2W+1 to avoid corner issues when casting/extending.
    logic signed [(2*W+1):0] prod_reg [N];
    logic [1:0]            mode2_op;
    logic                  mode2_elemwise;
    logic                  v2;
    logic                  clr2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v2             <= 1'b0;
            clr2           <= 1'b0;
            mode2_op       <= 2'd0;
            mode2_elemwise <= 1'b0;
        end else if (advance) begin
            v2             <= v1;
            clr2           <= clr1;
            mode2_op       <= mode1_op;
            mode2_elemwise <= mode1_elemwise;

            for (int i = 0; i < N; i++) begin
                unique case (mode1_op)
                    2'd0: begin // signed * signed
                        prod_reg[i] <= $signed(a_reg[i]) * $signed(b_reg[i]);
                    end
                    2'd1: begin // signed * unsigned
                        prod_reg[i] <= $signed(a_reg[i]) * $signed({1'b0, b_reg[i]});
                    end
                    default: begin // 2'd2 (unsigned*unsigned) and 2'd3 fallback
                        prod_reg[i] <= $signed({1'b0, a_reg[i]}) * $signed({1'b0, b_reg[i]});
                    end
                endcase
            end
        end
    end

    // ============================================================
    // Stage 3: reduction + vector pass-through
    // ============================================================
    logic signed [ACC_W-1:0] vec_sum_comb;
    logic signed [ACC_W-1:0] vec_sum_reg;
    logic signed [ACC_W-1:0] vec_pass_reg [N];
    logic                    mode3_elemwise;
    logic                    v3;
    logic                    clr3;

    always_comb begin
        vec_sum_comb = '0;
        for (int i = 0; i < N; i++) begin
            vec_sum_comb += $signed(prod_reg[i]);
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v3             <= 1'b0;
            clr3           <= 1'b0;
            mode3_elemwise <= 1'b0;
            vec_sum_reg    <= '0;
            for (int i = 0; i < N; i++) begin
                vec_pass_reg[i] <= '0;
            end
        end else if (advance) begin
            v3             <= v2;
            clr3           <= clr2;
            mode3_elemwise <= mode2_elemwise;
            vec_sum_reg    <= vec_sum_comb;

            for (int i = 0; i < N; i++) begin
                vec_pass_reg[i] <= $signed(prod_reg[i]);
            end
        end
    end

    // ============================================================
    // Stage 4: unified output
    // ============================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid      <= 1'b0;
            out_valid_mask <= '0;
            for (int i = 0; i < N; i++) begin
                out_vec[i] <= '0;
            end
        end else if (advance) begin
            out_valid <= v3;

            if (v3) begin
                if (mode3_elemwise) begin
                    // =========================================
                    // ELEMWISE MODE
                    // =========================================
                    out_valid_mask <= {N{1'b1}};
                    for (int i = 0; i < N; i++) begin
                        out_vec[i] <= vec_pass_reg[i];
                    end
                end else begin
                    // =========================================
                    // DOT MODE
                    // =========================================
                    out_valid_mask <= {{(N-1){1'b0}}, 1'b1};

                    if (clr3)
                        out_vec[0] <= vec_sum_reg;
                    else
                        out_vec[0] <= out_vec[0] + vec_sum_reg;

                    for (int i = 1; i < N; i++) begin
                        out_vec[i] <= '0;
                    end
                end
            end
        end
    end

endmodule