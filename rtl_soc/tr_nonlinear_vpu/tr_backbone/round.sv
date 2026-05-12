///////////////////////////////////////////////////////////////////////////////////////////////////////
// UPDATE:                                                                                           //
//      is_zero is assigned the rounded_mag value, instead of the input x.                           //
//      The output fliped_rounded_int cleanly gets ~rounded_mag[LUT_IDX_W-1:0], removing the is_zero check.    //
///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
// UPDATE:
//      Two modes created: 
//      MODE 1 - The original q4.4 rounding for the 8-bit LUT.
//      MODE 2 - 12-bit signed LUT for LayerNorm.
//////////////////////////////////////////////////////////////////
/*
 * @module   round
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    WIDTH           TODO: Add description
 * @param    FRAC_W          TODO: Add description
 * @param    LUT_IDX_W       TODO: Add description
 */
module round #(
    parameter int WIDTH = 8,
    parameter int FRAC_W = 4,
    parameter int LUT_IDX_W = 3
)(
    input  logic signed [WIDTH-1:0] x,           
    output logic is_zero,
    output logic is_ceil,
    output logic [LUT_IDX_W-1:0] lut_idx  // Index for LUT
);

    localparam int INT_W = WIDTH - FRAC_W;
    
    // 1. Extraction
    wire signed [INT_W-1:0] trunc_int;      // The integer part
    wire                    frac_round_bit; // The 0.5 fractional bit
    wire signed [INT_W-1:0] rounded_mag;

    // 2. Round-to-Nearest (Toward Zero for negatives)
    // If fractional bit is 1 (e.g., -1.5), we add 1 to the negative number to get -1.0

    assign trunc_int        = x[WIDTH-1:FRAC_W];
    assign frac_round_bit   = x[FRAC_W-1];
    assign is_ceil          = frac_round_bit;
    assign rounded_mag      = (frac_round_bit) ? (trunc_int + 'b1) : trunc_int;

    // 3. Zero Detection
    assign is_zero          = (rounded_mag == '0);
    
    // 4. LUT Index Generation
    generate
        if (WIDTH == 8 && FRAC_W == 4) begin : gen_idx_8bit
            // ----------------------------------------------------------------
            // MODE 1: 8-bit Negative-Only LUT Indexing
            // ----------------------------------------------------------------
            // Use bitwise NOT to map negative magnitudes to 0-based index 
            assign lut_idx = ~rounded_mag[LUT_IDX_W-1:0];
        end else begin : gen_idx_12bit
            // ----------------------------------------------------------------
            // MODE 2: 12-bit Signed LUT Indexing (LayerNorm)
            // ----------------------------------------------------------------
            // LayerNorm requires a signed LUT (spanning positive and negative anchors).
            // We just pass the raw, rounded integer out as the index.
            // The LUT module will handle the signed indexing.
            assign lut_idx = rounded_mag[LUT_IDX_W-1:0];
        end
    endgenerate

endmodule