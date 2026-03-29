module q12_8_round_neg (
    input  wire signed [19:0] x,                 // Q12.8 signed input
    output wire               is_zero,           // zero detector
    output wire               is_ceil,           // zero detector
    output wire        [3:0]  fliped_rounded_int // Index for LUT (0 to 7)
);

    // 1. Extraction
    wire signed [11:0] trunc_int;   // The integer part
    wire frac_round_bit;            // The 0.5 fractional bit
    wire signed [11:0] rounded_mag;

    // 2. Round-to-Nearest (Toward Zero for negatives)
    // If fractional bit is 1 (e.g., -1.5), we add 1 to the negative number to get -1.0

    assign trunc_int        = x[19:8];
    assign frac_round_bit   = x[7];
    assign is_ceil          = frac_round_bit;
    assign rounded_mag      = (frac_round_bit) ? (trunc_int + 'b1) : trunc_int;

    // 3. Zero Detection
    assign is_zero          = (rounded_mag == '0);
    
    // 4. Flipping for LUT Index
    // LayerNorm requires a signed LUT (spanning positive and negative anchors).
    // We just pass the raw, rounded integer out as the index.
    // The LUT module will handle the signed indexing.
    assign fliped_rounded_int = rounded_mag[3:0];

endmodule