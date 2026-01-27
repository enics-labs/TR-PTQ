module requant_unit (
    input  logic        clk,
    input  logic        rst_n,
    
    // Handshake
    input  logic        in_valid,
    output logic        in_ready,
    input  logic        out_ready,
    output logic        out_valid,

    // Data
    input  signed [31:0] acc_sum,
    input  signed [31:0] m_0,
    input  signed [4:0]  f_shift,
    input  signed [31:0] bias,
    output logic signed [7:0] out_quant
);

    // --- Pipeline Stage 1 Registers ---
    logic signed [63:0] prod_reg;
    logic signed [4:0]  f_shift_p1;
    logic signed [31:0] bias_p1;
    logic               val_p1;

    // --- Pipeline Stage 2 Registers ---
    logic signed [31:0] shifted_val_reg;
    logic signed [31:0] bias_p2;
    logic               val_p2;

    // ------------------------------------------------------------
    // Handshake Logic
    // ------------------------------------------------------------
    // We can accept new data if the next stage is ready or will be empty
    // To keep it simple and high-performance, we use the downstream out_ready
    assign in_ready = (!val_p1 && !val_p2) || out_ready;

    // ------------------------------------------------------------
    // Stage 1: Multiplication
    // ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            val_p1 <= 1'b0;
        end else if (in_ready) begin
            val_p1      <= in_valid;
            prod_reg    <= acc_sum * m_0;
            f_shift_p1  <= f_shift;
            bias_p1     <= bias;
        end
    end

    // ------------------------------------------------------------
    // Stage 2: Rounding & Shifting (The new "Cut")
    // ------------------------------------------------------------
    // Breaking the path between the 64-bit rounding and the bias adder
    logic signed [31:0] scaled_comb;
    assign scaled_comb = (prod_reg + 64'sh40000000) >>> 31;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            val_p2 <= 1'b0;
        end else if (out_ready || !val_p2) begin
            val_p2           <= val_p1;
            shifted_val_reg  <= scaled_comb >>> f_shift_p1;
            bias_p2          <= bias_p1;
        end
    end

    // ------------------------------------------------------------
    // Stage 3: Bias & Saturation
    // ------------------------------------------------------------
    logic signed [31:0] biased_val;
    assign biased_val = shifted_val_reg + bias_p2;

    assign out_valid = val_p2;

    always_comb begin
        if (biased_val > 32'sd127)
            out_quant = 8'sd127;
        else if (biased_val < -32'sd128)
            out_quant = -8'sd128;
        else
            out_quant = biased_val[7:0];
    end

endmodule