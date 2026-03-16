// ===================================================================================
// ARCHITECTURE NOTE: Mixed-Sign Multiplication & The Zero-Padding Bug
// ===================================================================================
// This module supports mixed-sign vector dot products via the `op_mode` signal.
//
// ORIGINAL BUG:
// Originally, all operands were prepended with a zero: `$signed({1'b0, a_reg})`.
// While this is required for unsigned numbers, doing this to a negative signed 
// number destroys its sign bit. For example, an 8-bit -1 (1111_1111) becomes a 
// 9-bit +255 (0_1111_1111). This caused massive positive calculation errors.
//
// THE FIX:
// We zero-pad ONLY the unsigned operands to prevent their MSB from being
// accidentally interpreted as a negative two's complement sign, while 
// leaving signed operands untouched so they sign-extend naturally.
//
//   * Mode 0 (SS) - Signed x Signed: 
//       Both operands are cast directly to $signed(). SV naturally sign-extends.
//       Logic: $signed(a) * $signed(b)
//
//   * Mode 1 (SU) - Signed x Unsigned: 
//       Operand 'a' is signed (left alone). Operand 'b' is unsigned, so we 
//       force a leading zero to ensure it remains a positive magnitude.
//       Logic: $signed(a) * $signed({1'b0, b})
//
//   * Mode 2 (UU) - Unsigned x Unsigned:
//       Both operands are zero-padded to protect their MSBs, then cast to signed
//       so the resulting product container behaves correctly in the accumulator.
//       Logic: $signed({1'b0, a}) * $signed({1'b0, b})
// ===================================================================================
module vec_mac_dsp48 #(
    parameter int N     = 8,
    parameter int W     = 18,
    parameter int ACC_W = 48
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Input stream
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic [1:0]              op_mode,   // 0:SS, 1:SU, 2:UU
    input  logic [W-1:0]            a [N],      // raw bits
    input  logic [W-1:0]            b [N],      // raw bits
    input  logic                    clear_acc,

    // Output stream
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic signed [ACC_W-1:0] out_dot
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
    logic [1:0]   mode1;
    logic         v1;
    logic         clr1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v1    <= 1'b0;
            clr1  <= 1'b0;
            mode1 <= 2'd0;
        end else if (advance) begin
            v1    <= in_valid;
            clr1  <= clear_acc;
            mode1 <= op_mode;
            for (int i = 0; i < N; i++) begin
                a_reg[i] <= a[i];
                b_reg[i] <= b[i];
            end
        end
    end

    // ============================================================
    // Stage 2: registered products (mixed signedness)
    // ============================================================
    // Use signed products so sign-extension is correct downstream.
    // Width: 2W+1 to avoid corner issues when casting/extending.
    logic signed [(2*W):0] prod_reg [N];
    logic [(2*W):0] prod_reg2 [N];
    logic [1:0]            mode2;
    logic                  v2;
    logic                  clr2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v2    <= 1'b0;
            clr2  <= 1'b0;
            mode2 <= 2'd0;
        end else if (advance) begin
            v2    <= v1;
            clr2  <= clr1;
            mode2 <= mode1;

            for (int i = 0; i < N; i++) begin
                unique case (mode1)
                    2'd0: begin // signed * signed
                        // prod_reg[i] <= (2*W)'($signed({1'b0, a_reg[i]}) * $signed({1'b0, b_reg[i]}));
                        prod_reg[i] <= $signed(a_reg[i]) * $signed(b_reg[i]);
                    end
                    2'd1: begin // signed * unsigned
                        // prod_reg[i] <= (2*W)'($signed({1'b0, a_reg[i]}) * $signed({1'b0, $unsigned(b_reg[i])}));
                        prod_reg[i] <= $signed(a_reg[i]) * $signed({1'b0, b_reg[i]});
                    end
                    default: begin // 2'd2 (unsigned*unsigned) and 2'd3 fallback
                        // Cast to unsigned magnitude then into signed container (non-negative)
                        // prod_reg[i] <= $signed({1'b0, (2*W)'($unsigned(a_reg[i]) * $unsigned(b_reg[i]))});
                        prod_reg[i] <= $signed({1'b0, a_reg[i]}) * $signed({1'b0, b_reg[i]});
                    end
                endcase
            end
        end
    end

    // ============================================================
    // Stage 3: reduction (sign-extend products into ACC_W)
    // ============================================================
    logic signed [ACC_W-1:0] vec_sum_comb;
    logic signed [ACC_W-1:0] vec_sum_reg;
    logic                    v3;
    logic                    clr3;

    always_comb begin
        vec_sum_comb = '0;
        for (int i = 0; i < N; i++) begin
            // prod_reg[i] is signed, so this is a true sign-extend into ACC_W
            vec_sum_comb += $signed(prod_reg[i]);
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v3          <= 1'b0;
            clr3        <= 1'b0;
            vec_sum_reg <= '0;
        end else if (advance) begin
            v3          <= v2;
            clr3        <= clr2;
            vec_sum_reg <= vec_sum_comb;
        end
    end

    // ============================================================
    // Stage 4: accumulation into output buffer
    // ============================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_dot   <= '0;
        end else if (advance) begin
            out_valid <= v3;
            if (v3) begin
                if (clr3) out_dot <= vec_sum_reg;
                else      out_dot <= out_dot + vec_sum_reg;
            end
        end
    end

endmodule
