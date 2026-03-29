module tr_ln #(
    parameter int WIDTH = 16,
    parameter int BITS  = 4,        // Fractional bits
    parameter int OUT_WIDTH = 8     // Defaults to SoftMax/GELU 16->8 reduction
)(
    input  wire        [WIDTH-1:0]   xq,
    output wire signed [OUT_WIDTH-1:0] yq
);

    // def new_ln(xq, bits):
    //     aq = xq.log2().floor().int() - bits
    //     # compute xq >> aq if aq > 0 else xq << -aq)
    //     k1 = safe_shift_r(xq, aq)
    //     # compute 2**(-aq))*xq
    //     k2 = ((aq-1) << bits)
    //     # compute (2**(-aq))*xq + ((aq-1)*2**bits)
    //     k = (k1 + k2)
    //     # yq = ln2q * ((2**(-aq))*xq + ((aq-1)*2**bits))
    //     yq = (((k>>1)+(k>>3)+(k>>4)))
    //     return yq

    // Use 32-bit signed variables for all internal math to prevent simulator truncation bugs
    logic [5:0]         msb;
    logic [2:0] aq_full;
    logic signed [7:0] k1_full;
    logic signed [7:0] k2_full;
    logic signed [8:0] k_full;
    logic signed [7:0] yq_full;

always_comb begin
        // 1. Find the MSB (log2 floor)
        //     aq = xq.log2().floor().int() - bits
        msb = 0;
        for (int i = 0; i < WIDTH; i++) begin
            if (xq[i]) msb = i;
        end
        // 0 < a < 5.6
        // 2. Calculate aq = msb - BITS
        aq_full = $signed({1'b0, msb}) - $signed(BITS);
        k1_full = xq >> aq_full;
        // 3. Perform the safe shift
        //     k1 = safe_shift_r(xq, aq)
        // if (aq_full > 0) begin
        //     k1_full = xq >> aq_full;
        // end else begin
        //     k1_full = xq << (-aq_full);
        // end

        // 4. Compute k2 = (aq - 1) * (1 << BITS)
        k2_full = ($signed({1'b0, aq_full}) - 1) <<< BITS;

        // 5. Sum them up (Safe signed addition)
        k_full = k1_full + k2_full;

        // 6. Final approximation: yq = (k/2) + (k/8) + (k/16)
        // yq_full = (k_full * 11) >>> 4;
        yq_full = (k_full >>> 1) + (k_full >>> 3) + (k_full >>> 4);

    end
    
    // Assign back out to the parameterized width
    assign yq = yq_full[OUT_WIDTH-1:0];

endmodule