/*
 * @module   tr_ln_alu
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    WIDTH           TODO: Add description
 * @param    BITS            TODO: Add description
 * @param    OUT_WIDTH       TODO: Add description
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