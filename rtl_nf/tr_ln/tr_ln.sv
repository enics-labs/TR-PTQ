/*
 * @module   tr_ln
 * @brief    Fast Base-e Logarithm Approximation
 * @details  Calculates ln(x) dynamically by extracting the Most Significant Bit 
 *           (log2 floor) and applying fixed-point shift approximations.
 */
`timescale 1ns/1ps

module tr_ln #(
    parameter int WIDTH     = 16,
    parameter int BITS      = 4,  // Fractional bits
    parameter int OUT_WIDTH = 8   // Target width for output packing
)(
    input  wire        [WIDTH-1:0]     xq,
    output wire signed [OUT_WIDTH-1:0] yq
);

    // ========================================================================
    // Dynamic Bit-Width Calculations
    // ========================================================================
    localparam int MSB_W = $clog2(WIDTH);        
    localparam int AQ_W  = MSB_W + 1;            
    localparam int K1_W  = BITS + 2;             
    localparam int K_W   = AQ_W + BITS;          

    logic [MSB_W-1:0]       msb;
    logic signed [AQ_W-1:0] aq_full;

    logic signed [K1_W-1:0] k1_full;
    logic signed [K_W-1:0]  k2_full;
    logic signed [K_W-1:0]  k_full;
    logic signed [K_W-1:0]  yq_full;

    logic [WIDTH-1:0]       normalized_x;

    // ========================================================================
    // Approximation Pipeline
    // ========================================================================
    always_comb begin
        // 1. Find the MSB (log2 floor)
        msb = '0;
        for (int i = 0; i < WIDTH; i++) begin
            if (xq[i]) msb = i;
        end

        // 2. Calculate aq = msb - BITS
        aq_full = $signed({1'b0, msb}) - $signed(AQ_W'(BITS));

        // 3. Normalize input and extract fractional base
        normalized_x = xq << (WIDTH - 1 - msb);
        k1_full = $signed({1'b0, normalized_x[WIDTH-1 : WIDTH-1-BITS]});

        // 4. Compute k2 = (aq - 1) * (1 << BITS)
        k2_full = $signed(aq_full - 1) <<< BITS;

        // 5. Sum Base and Normalization (Safe signed addition)
        k_full = k1_full + k2_full;

        // 6. Final Base-e Conversion: yq = (k/2) + (k/8) + (k/16)
        yq_full = (k_full >>> 1) + (k_full >>> 3) + (k_full >>> 4);
    end
    
    // Assign back out to the parameterized target width
    assign yq = OUT_WIDTH'(yq_full);

endmodule