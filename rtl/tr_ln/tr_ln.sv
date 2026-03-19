module tr_ln #(
    parameter int WIDTH = 16,
    parameter int BITS  = 4,        // Fractional bits
    parameter int OUT_WIDTH = 8     // Defaults to SoftMax/GELU 16->8 reduction
)(
    input  wire        [WIDTH-1:0]   xq,
    output wire signed [OUT_WIDTH-1:0] yq
);

    // Use 32-bit signed variables for all internal math to prevent simulator truncation bugs
    logic [5:0]         msb;
    logic signed [31:0] aq_full;
    logic signed [31:0] k1_full;
    logic signed [31:0] k2_full;
    logic signed [31:0] k_full;
    logic signed [31:0] yq_full;

always_comb begin
        // 1. Find the MSB (log2 floor)
        msb = 0;
        for (int i = 0; i < WIDTH; i++) begin
            if (xq[i]) msb = i;
        end

        // 2. Calculate aq = msb - BITS
        aq_full = $signed({1'b0, msb}) - $signed(BITS);

        // 3. Perform the safe shift
        if (aq_full > 0) begin
            k1_full = xq >> aq_full;
        end else begin
            k1_full = xq << (-aq_full);
        end

        // 4. Compute k2 = (aq - 1) * (1 << BITS)
        k2_full = (aq_full - 1) * (1 << BITS);

        // 5. Sum them up (Safe signed addition)
        k_full = k1_full + k2_full;

        // 6. Final approximation: yq = (k/2) + (k/8) + (k/16)
        yq_full = (k_full * 11) >>> 4;
    end
    
    // Assign back out to the parameterized width
    assign yq = yq_full[OUT_WIDTH-1:0];

endmodule