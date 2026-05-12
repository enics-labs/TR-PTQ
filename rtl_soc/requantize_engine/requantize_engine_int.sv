/*
 * @module   requantize_array_engine
 * @brief    Vectorized Integer Requantizer with Saturation.
 * @details  Converts wide accumulators back to base precision using:
 *           Out = Saturate( (Acc * M + Bias) >>> S )
 *           Features a 3-stage pipeline for high-frequency timing closure.
 */
module requantize_engine_int #(
    parameter int N       = 16, // Vector dimension
    parameter int ACC_W   = 32, // Input accumulator width
    parameter int MUL_W   = 32, // Multiplier scale width
    parameter int SHIFT_W = 6,  // Shift amount width (up to 63 bits)
    parameter int OUT_W   = 8   // Target output width (e.g., 8-bit)
)(
    input  logic clk,
    input  logic rst_n,

    // Stream Handshake
    input  logic in_valid,
    output logic in_ready,

    // Datapath Inputs
    input  logic signed [ACC_W-1:0]  acc_in [N],
    input  logic signed [MUL_W-1:0]  multiplier, // M from Controller
    input  logic [SHIFT_W-1:0]       shift,      // S from Controller

    // Stream Output
    output logic out_valid,
    input  logic out_ready,
    output logic signed [OUT_W-1:0]  out_vec [N]
);

    // Dynamic wide product width
    localparam int PROD_W = ACC_W + MUL_W; 
    
    // Saturation Bounds for the target OUT_W
    localparam signed [PROD_W-1:0] MAX_VAL =  (1 << (OUT_W - 1)) - 1;
    localparam signed [PROD_W-1:0] MIN_VAL = -(1 << (OUT_W - 1));

    logic advance;
    assign advance = (~out_valid) || (out_valid && out_ready);
    assign in_ready = advance;

    // ============================================================
    // Stage 1: Wide Multiplication (Acc * M)
    // ============================================================
    logic signed [PROD_W-1:0]  prod_reg [N];
    logic [SHIFT_W-1:0]        shift_reg1;
    logic                      v1;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            shift_reg1 <= '0;
            for (int i = 0; i < N; i++) prod_reg[i] <= '0;
        end else if (advance) begin
            v1 <= in_valid;
            shift_reg1 <= shift; // Pipeline the shift parameter
            for (int i = 0; i < N; i++) begin
                // Fully signed multiplication expanding to PROD_W
                prod_reg[i] <= $signed(acc_in[i]) * $signed(multiplier);
            end
        end
    end

    // ============================================================
    // Stage 2: Rounding Bias + Arithmetic Right Shift
    // ============================================================
    logic signed [PROD_W-1:0] shifted_reg [N];
    logic                     v2;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            v2 <= 1'b0;
            for (int i = 0; i < N; i++) shifted_reg[i] <= '0;
        end else if (advance) begin
            v2 <= v1;
            for (int i = 0; i < N; i++) begin
                // Safe generation of the rounding bias: 1 << (S - 1)
                // If shift is 0, bias is 0.
                logic signed [PROD_W-1:0] bias;
                bias = (shift_reg1 > 0) ? (PROD_W'(1) << (shift_reg1 - 1)) : '0;
                
                // Add bias, then arithmetic right shift (>>> preserves sign bit)
                shifted_reg[i] <= (prod_reg[i] + bias) >>> shift_reg1;
            end
        end
    end

    // ============================================================
    // Stage 3: Saturation Clamping
    // ============================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            for (int i = 0; i < N; i++) out_vec[i] <= '0;
        end else if (advance) begin
            out_valid <= v2;
            if (v2) begin
                for (int i = 0; i < N; i++) begin
                    if (shifted_reg[i] > MAX_VAL) begin
                        // Clamp to ceiling (e.g., +127)
                        out_vec[i] <= OUT_W'(MAX_VAL); 
                    end else if (shifted_reg[i] < MIN_VAL) begin
                        // Clamp to floor (e.g., -128)
                        out_vec[i] <= OUT_W'(MIN_VAL); 
                    end else begin
                        // Safe to cast down
                        out_vec[i] <= OUT_W'(shifted_reg[i]);
                    end
                end
            end
        end
    end
    
endmodule