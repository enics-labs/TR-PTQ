/*
 * @module   round
 * @brief    Rounding Logic for Taylor-Region LUT Indexing
 * @details  Supports Negative-Only indexing (8-bit) and full Signed 
 *           indexing (12-bit) dynamically based on parameter synthesis.
 */
`timescale 1ns/1ps

module round #(
    parameter int WIDTH     = 8,
    parameter int FRAC_W    = 4,
    parameter int LUT_IDX_W = 3
)(
    input  wire signed [WIDTH-1:0] x,           
    output wire                    is_zero,
    output wire                    is_ceil,
    output wire    [LUT_IDX_W-1:0] lut_idx 
);

    localparam int INT_W = WIDTH - FRAC_W;
    
    // ========================================================================
    // 1. Bit Extraction
    // ========================================================================
    wire signed [INT_W-1:0] trunc_int;      
    wire                    frac_round_bit; 
    wire signed [INT_W-1:0] rounded_mag;

    assign trunc_int      = x[WIDTH-1:FRAC_W];
    assign frac_round_bit = x[FRAC_W-1];
    assign is_ceil        = frac_round_bit;

    // ========================================================================
    // 2. Round-to-Nearest 
    // ========================================================================
    // Fractional bit represents 0.5. Adds 1 to magnitude if present.
    assign rounded_mag = (frac_round_bit) ? (trunc_int + 'b1) : trunc_int;

    // ========================================================================
    // 3. Flags and LUT Mapping
    // ========================================================================
    assign is_zero = (rounded_mag == '0);
    
    generate
        if (WIDTH == 8 && FRAC_W == 4) begin : gen_idx_8bit
            // ---------------------------------------------------------
            // MODE 1: 8-bit Negative-Only LUT Indexing
            // ---------------------------------------------------------
            // Use bitwise NOT to map negative magnitudes to 0-based index 
            assign lut_idx = ~rounded_mag[LUT_IDX_W-1:0];
            
        end else begin : gen_idx_12bit
            // ---------------------------------------------------------
            // MODE 2: 12-bit Signed LUT Indexing (LayerNorm)
            // ---------------------------------------------------------
            // Direct pass-through for signed multi-domain anchors
            assign lut_idx = rounded_mag[LUT_IDX_W-1:0];
        end
    endgenerate

endmodule