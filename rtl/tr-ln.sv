module tr_ln #(
    parameter WIDTH = 16,
    parameter BITS  = 4  // Fractional bits
)(
    input  wire [WIDTH-1:0] xq,
    output wire [WIDTH/2-1:0] yq
);

    reg [5:0] msb;
    integer i;

    // 1. Find the MSB (log2 floor)
    // This replicates xq.log2().floor().int()
    always @(*) begin
        msb = 0;
        for (i = 0; i < WIDTH; i = i + 1) begin
            if (xq[i]) msb = i;
        end
    end

    // 2. Calculate aq = msb - bits
    // We use a signed wire to handle negative shifts
    wire signed [7:0] aq = msb - BITS;

    // 3. Perform the safe shift (k1 = xq >> aq)
    wire [WIDTH-1:0] k1;
    assign k1 = (aq > 0) ? (xq >> aq) : (xq << (-aq));

    // 4. Compute k2 = (aq - 1) << bits
    wire signed [WIDTH-1:0] k2 = (aq - 1) <<< BITS;

    // 5. Sum them up
    wire signed [WIDTH-1:0] k = k1 + k2;

    // 6. Final approximation: yq = (k/2) + (k/8) + (k/16)
    // This represents the ln(2) multiplication
    assign yq = (k >>> 1) + (k >>> 3) + (k >>> 4);

endmodule