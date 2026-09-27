/*
 * @module   tr_ln_alu
 * @brief    Integer natural-logarithm approximator via MSB-exponent +
 *           linear-mantissa log2 decomposition.
 * @details  Finds the position of xq's MSB (floor(log2(xq))), normalizes xq
 *           so that bit sits at the top of the word, and takes its top
 *           BITS+1 bits as a linear mantissa term k1 (log2(1.f) ~= f over
 *           [0,1), the standard piecewise-linear log2 approximation).
 *           Combines that with the (fixed-point-scaled) exponent term
 *           k2=(msb-BITS-1)<<<BITS into a single log2(xq) estimate k=k1+k2,
 *           then converts log2->ln by multiplying by the fixed-point
 *           constant 0.6875 (~= ln(2), realized as k>>>1 + k>>>3 + k>>>4 to
 *           avoid a real multiplier), and saturates the result to
 *           OUT_WIDTH. xq==0 short-circuits to yq=0.
 *
 * @param    WIDTH      Input width (xq).
 * @param    BITS       Fractional bits of xq (also the mantissa's linear-term width).
 * @param    OUT_WIDTH  Output width (yq), saturated.
 */
module tr_ln_alu #(
    parameter int WIDTH = 16,
    parameter int BITS  = 4,        // Fractional bits
    parameter int OUT_WIDTH = 8     // Defaults to SoftMax/GELU 16->8 reduction
)(
    input  logic [WIDTH-1:0]   xq,
    output logic signed [OUT_WIDTH-1:0] yq
);

    // ========================================================================
    // Dynamic Bit-Width Calculations
    // ========================================================================
    localparam int MSB_W = $clog2(WIDTH);        // e.g., 4 for W=16 | 5 for W=20
    localparam int AQ_W  = MSB_W + 1;            // e.g., 5 for W=16 | 6 for W=20
    localparam int K1_W  = BITS + 2;             // e.g., 6 for B=4
    localparam int K_W   = AQ_W + BITS;          // e.g., 9 for W=16 | 10 for W=20

    logic [MSB_W-1:0]         msb;
    logic signed [AQ_W-1:0]   aq_full;

    logic signed [K1_W-1:0] k1_full;
    logic signed [K_W-1:0] k2_full;
    logic signed [K_W-1:0]   k_full;
    logic signed [K_W-1:0]   yq_full;

    logic [WIDTH-1:0]       normalized_x;

    always_comb begin
        // Defaults so every variable is assigned on all paths (no inferred
        // latches; Quartus treats always_comb latch inference as a hard error).
        msb          = '0;
        aq_full      = '0;
        normalized_x = '0;
        k1_full      = '0;
        k2_full      = '0;
        k_full       = '0;
        yq_full      = '0;

        if (xq == 0) begin
            yq_full = '0;
        end else begin
            // 1. Find the MSB (log2 floor)
            msb = '0;
            for (int i = 0; i < WIDTH; i++) begin
                if (xq[i]) msb = i;
            end

            // 2. Calculate aq = msb - BITS
            aq_full = $signed({1'b0, msb}) - $signed(AQ_W'(BITS));

            // 3. Perform the shift
            normalized_x = xq << (WIDTH - 1 - msb);
            k1_full = $signed({1'b0, normalized_x[WIDTH-1 : WIDTH-1-BITS]});

            // 4. Compute k2 = (aq - 1) * (1 << BITS)
            k2_full = $signed(aq_full - 1) <<< BITS;

            // 5. Sum them up (Safe signed addition)
            k_full = k1_full + k2_full;

            // 6. Final approximation: yq = (k/2) + (k/8) + (k/16)
            yq_full = (k_full >>> 1) + (k_full >>> 3) + (k_full >>> 4);
        end
    end

    always_comb begin
        if (yq_full > (2**(OUT_WIDTH-1) - 1))
            yq = (2**(OUT_WIDTH-1) - 1);
        else if (yq_full < -(2**(OUT_WIDTH-1)))
            yq = -(2**(OUT_WIDTH-1));
        else
            yq = OUT_WIDTH'(yq_full);
    end

endmodule