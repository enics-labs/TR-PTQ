module vec_mac_dsp48 #(
    parameter int N     = 8,    // vector length
    parameter int W     = 18,   // signed lane width (DSP48-friendly)
    parameter int ACC_W = 48    // accumulator width (DSP48 P width)
)(
    input  logic                    clk,
    input  logic                    rst_n,       // active-low synchronous reset

    // Input stream (vector per beat)
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic signed [W-1:0]     a [N],
    input  logic signed [W-1:0]     b [N],
    input  logic                    clear_acc,  // assert with the first beat of a new accumulation window

    // Output stream
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic signed [ACC_W-1:0] out_dot
);

    // ============================================================
    // Handshake / pipeline control
    // ============================================================
    // We implement a 1-entry output buffer (out_valid/out_dot).
    // The pipeline advances only when the output buffer can accept
    // a new result (either empty, or being consumed this cycle).
    //
    // This ensures correctness under backpressure.
    // ============================================================
    logic advance;
    assign advance  = (~out_valid) || (out_valid && out_ready);
    assign in_ready = advance;

    // ============================================================
    // Stage 1: AREG/BREG (input registers)
    // ============================================================
    logic signed [W-1:0] a_reg [N];
    logic signed [W-1:0] b_reg [N];
    logic                v1;
    logic                clr1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v1   <= 1'b0;
            clr1 <= 1'b0;
        end else if (advance) begin
            v1   <= in_valid;
            clr1 <= clear_acc;

            for (int i = 0; i < N; i++) begin
                a_reg[i] <= a[i];
                b_reg[i] <= b[i];
            end
        end
    end

    // ============================================================
    // Stage 2: MREG (registered products)
    // ============================================================
    logic signed [(2*W)-1:0] prod_reg [N];
    logic                    v2;
    logic                    clr2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v2   <= 1'b0;
            clr2 <= 1'b0;
        end else if (advance) begin
            v2   <= v1;
            clr2 <= clr1;

            for (int i = 0; i < N; i++) begin
                // DSP48 inference-friendly: registered multiply
                prod_reg[i] <= (2*W)'(a_reg[i]) * (2*W)'(b_reg[i]);
            end
        end
    end

    // ============================================================
    // Stage 3: Vector reduction (registered)
    // ============================================================
    logic signed [ACC_W-1:0] vec_sum_comb;
    logic signed [ACC_W-1:0] vec_sum_reg;
    logic                    v3;
    logic                    clr3;

    // Combinational reduction (adder tree is inferred here).
    // Keeping it separate from always_ff improves synthesis stability.
    always_comb begin
        vec_sum_comb = '0;
        for (int i = 0; i < N; i++) begin
            // Sign-extend product to ACC_W before accumulation
            vec_sum_comb += {{(ACC_W-(2*W)){prod_reg[i][(2*W)-1]}}, prod_reg[i]};
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v3         <= 1'b0;
            clr3       <= 1'b0;
            vec_sum_reg<= '0;
        end else if (advance) begin
            v3          <= v2;
            clr3        <= clr2;
            vec_sum_reg <= vec_sum_comb;
        end
    end

    // ============================================================
    // Stage 4: DSP48-style accumulation into output buffer (PREG)
    // ============================================================
    // out_dot acts as the P register (accumulator).
    //
    // Behavior:
    // - If clr3 asserted with a valid beat: out_dot := vec_sum_reg
    // - Else if valid beat: out_dot := out_dot + vec_sum_reg
    // - out_valid asserts when a new accumulator value is produced
    //
    // Note: This produces one output per input beat (after latency),
    // carrying the running accumulated dot-product.
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

