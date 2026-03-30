///////////////////////////////////////////////////////////////////////////////////////////////////////
// UPDATE:                                                                                           //
//      is_zero is assigned the rounded_mag value, instead of the input x.                           //
//      The output fliped_rounded_int cleanly gets ~rounded_mag[2:0], removing the is_zero check.    //
///////////////////////////////////////////////////////////////////////////////////////////////////////
module q4_4_round_neg (
    input  wire signed [7:0] x,           // Q4.4 signed input (range: -8.0 to 0.0)
    output wire              is_zero,     // zero detector
    output wire              is_ceil,     // zero detector
    output wire        [2:0] fliped_rounded_int // Index for LUT (0 to 7)
);

    // 1. Extraction
    wire signed [3:0] trunc_int; // The integer part
    wire frac_round_bit;   // The 0.5 fractional bit
    wire signed [3:0] rounded_mag;

    // 2. Round-to-Nearest (Toward Zero for negatives)
    // If fractional bit is 1 (e.g., -1.5), we add 1 to the negative number to get -1.0

    assign trunc_int        = x[7:4];
    assign frac_round_bit   = x[3];
    assign is_ceil          = frac_round_bit;
    assign rounded_mag      = (frac_round_bit) ? (trunc_int + 4'sd1) : trunc_int;

    // 3. Zero Detection
    assign is_zero          = (rounded_mag == 4'sd0);
    
    // 4. Flipping for LUT Index
    // We use ~ to turn negative integers into a 0-based magnitude index
    // Note: We only need the lower 3 bits since max magnitude is 8 (which flips to 7)
    assign fliped_rounded_int = ~rounded_mag[2:0];

endmodule