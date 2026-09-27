/*
 * @module   round
 * @brief    Rounds a fixed-point input to its nearest integer anchor and
 *           generates the corresponding shared-LUT index.
 * @details  Rounds x to the nearest integer (toward zero for negatives, via
 *           its 0.5 fractional bit), returning is_ceil (whether that
 *           rounding moved x up past its anchor -- the sign of the
 *           residual delta=x-anchor that tr_exp_alu's Taylor polynomial
 *           needs) and is_zero (the rounded integer is exactly 0). lut_idx
 *           is generated in one of two modes (DECAY_ONLY_LUT): MODE 1
 *           (default) bitwise-NOTs the rounded magnitude to map the
 *           negative-only decay anchors (0, -1, -2, ...) onto a 0-based
 *           index for shared_lut_rom's 8-entry decay table, saturating to
 *           the highest index (fully-decayed anchor) once the magnitude
 *           exceeds what LUT_IDX_W bits can distinguish; MODE 2 passes the
 *           rounded integer straight through as a signed index, for a
 *           bidirectional (positive and negative anchors) LUT used by
 *           LayerNorm-style callers.
 *
 * @param    WIDTH           Input width (x).
 * @param    FRAC_W          Fractional bits of x.
 * @param    LUT_IDX_W       Width of the generated LUT index (lut_idx).
 * @param    DECAY_ONLY_LUT  1: MODE 1 (decay-only anchor table, tr_exp_alu's
 *                            default); 0: MODE 2 (bidirectional/LayerNorm).
 */
module round #(
    parameter int WIDTH = 8,
    parameter int FRAC_W = 4,
    parameter int LUT_IDX_W = 3,
    parameter bit DECAY_ONLY_LUT = 1  // 1: MODE1 (decay-only anchor table); 0: MODE2 (bidirectional/LayerNorm)
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
        if (DECAY_ONLY_LUT) begin : gen_idx_8bit
            // ----------------------------------------------------------------
            // MODE 1: 8-bit Negative-Only LUT Indexing
            // ----------------------------------------------------------------
            // Use bitwise NOT to map negative magnitudes to 0-based index.
            // rounded_mag in [-1,-2^LUT_IDX_W] uses all LUT_IDX_W bits
            // distinctly; beyond that it would alias back to a low (large
            // anchor) index instead of decaying further, so saturate to the
            // highest index (its anchor is 0, i.e. fully decayed) instead.
            localparam logic [LUT_IDX_W-1:0] LUT_MAX = {LUT_IDX_W{1'b1}};
            assign lut_idx = (rounded_mag < -(int'(LUT_MAX) + 1)) ? LUT_MAX : ~rounded_mag[LUT_IDX_W-1:0];
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