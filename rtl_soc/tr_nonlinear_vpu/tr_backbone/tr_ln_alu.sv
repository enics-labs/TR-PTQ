/*
 * @module   tr_ln_alu
 * @brief    Integer natural-logarithm approximator: MSB-exponent +
 *           piecewise-linear fit of ln(1+f) over the mantissa fraction.
 * @details  Finds the position of xq's MSB (floor(log2(xq))), normalizes xq
 *           so that bit sits at the top of the word, and takes the BITS bits
 *           below it as the mantissa fraction fk (f = fk / 2^BITS). The
 *           mantissa range [0,1) is split into SEGMENTS equal segments
 *           (segment = top log2(SEGMENTS) bits of fk), each with its own
 *           least-squares line of ln(1+f), constants quantized to COEF_BITS:
 *               aq = msb - BITS
 *               yq = (aq*L*2^BITS + A[seg]*fk + B[seg]*2^BITS + 2^(COEF_BITS-1)) >>> COEF_BITS
 *           with L=round(ln2*2^COEF_BITS), A/B=round(slope/intercept*2^COEF_BITS).
 *           The result is ln(xq/2^BITS) in Q.BITS and is saturated to
 *           OUT_WIDTH. xq==0 short-circuits to yq=0. All multiplies are by
 *           constants (shift-add trees); there is no general multiplier.
 *           Bit-exact to tools/models/vpu_ops.py::_ln_core (ln_mode="pwl");
 *           the constant tables below are re-derived and asserted by
 *           verification/ln_pwl/gen_ln_pwl.py.
 *
 * @param    WIDTH      Input width (xq).
 * @param    BITS       Fractional bits of xq (also the mantissa fraction width).
 * @param    OUT_WIDTH  Output width (yq), saturated.
 * @param    SEGMENTS   Number of PWL segments (1, 2 or 4).
 * @param    COEF_BITS  Fixed-point bits of the PWL constants (6, 8 or 10).
 */
module tr_ln_alu #(
    parameter int WIDTH = 16,
    parameter int BITS  = 4,        // Fractional bits
    parameter int OUT_WIDTH = 8,    // Defaults to the Softmax 16->8 reduction
    parameter int SEGMENTS  = 2,    // PWL segments (1, 2, 4)
    parameter int COEF_BITS = 8     // PWL constant precision (6, 8, 10)
)(
    input  logic [WIDTH-1:0]   xq,
    output logic signed [OUT_WIDTH-1:0] yq
);

    // ========================================================================
    // Dynamic Bit-Width Calculations
    // ========================================================================
    localparam int MSB_W = $clog2(WIDTH);        // e.g., 4 for W=16 | 5 for W=20
    localparam int AQ_W  = MSB_W + 1;            // e.g., 5 for W=16 | 6 for W=20
    localparam int LG_SEG = (SEGMENTS > 1) ? $clog2(SEGMENTS) : 0;

    // ========================================================================
    // PWL constants. Key = SEGMENTS*64 + COEF_BITS.
    // ========================================================================
    function automatic int coef_a(input int segs, input int c, input int idx);
        case (segs*64 + c)
            1*64+8:  coef_a = 175;
            2*64+8:  coef_a = (idx == 0) ? 206 : 147;
            4*64+8:  case (idx) 0: coef_a = 228; 1: coef_a = 186; 2: coef_a = 158; default: coef_a = 137; endcase
            1*64+6:  coef_a = 44;
            2*64+6:  coef_a = (idx == 0) ? 52 : 37;
            2*64+10: coef_a = (idx == 0) ? 826 : 588;
            default: coef_a = 0;
        endcase
    endfunction

    function automatic int coef_b(input int segs, input int c, input int idx);
        case (segs*64 + c)
            1*64+8:  coef_b = 12;
            2*64+8:  coef_b = (idx == 0) ? 4 : 32;
            4*64+8:  case (idx) 0: coef_b = 1; 1: coef_b = 11; 2: coef_b = 25; default: coef_b = 41; endcase
            1*64+6:  coef_b = 3;
            2*64+6:  coef_b = (idx == 0) ? 1 : 8;
            2*64+10: coef_b = (idx == 0) ? 15 : 129;
            default: coef_b = 0;
        endcase
    endfunction

    function automatic int coef_l(input int c);   // round(ln2 * 2^c)
        case (c)
            6:  coef_l = 44;
            8:  coef_l = 177;
            10: coef_l = 710;
            default: coef_l = 0;
        endcase
    endfunction

    generate
        if (coef_l(COEF_BITS) == 0 || coef_a(SEGMENTS, COEF_BITS, 0) == 0)
            $error("tr_ln_alu: no PWL constants for SEGMENTS=%0d COEF_BITS=%0d", SEGMENTS, COEF_BITS);
        if (BITS < LG_SEG)
            $error("tr_ln_alu: BITS must be >= log2(SEGMENTS)");
    endgenerate

    // ========================================================================
    // MSB finder: balanced binary-search tree (NUM_LEVELS = MSB_W levels),
    // not a linear bit-by-bit scan. A linear "for each bit, if set, msb=i"
    // scan synthesizes as a WIDTH-deep chain of priority muxes (each level
    // waiting on the previous), which was the dominant term in a measured
    // FPGA timing violation on this exact path. xq is conceptually
    // zero-padded up to WIDTH_P2 (the next power of two) so this works for
    // any WIDTH, not just powers of two -- the padding bits are always 0 and
    // can never win a group's OR, so they can't affect the real result.
    // Each level halves the search range: nz[L][g] is whether ANY bit in
    // that group's (2^L)-wide range is set, idx[L][g] is that group's own
    // MSB position (0-based within the group), built by OR-ing in one more
    // bit as we ascend (bit (L-1) set if the upper half won, clear if the
    // lower half won). Same tree-reduction technique already used to fix the
    // equivalent linear scans in ibert_rmsnorm.sv/ibert_softmax.sv.
    // ========================================================================
    localparam int WIDTH_P2   = 1 << MSB_W;   // next power of two >= WIDTH
    localparam int NUM_LEVELS = MSB_W;        // == log2(WIDTH_P2)

    logic             nz  [0:NUM_LEVELS][0:WIDTH_P2-1];
    logic [MSB_W-1:0] idx [0:NUM_LEVELS][0:WIDTH_P2-1];

    genvar gl, gj;
    generate
        for (gj = 0; gj < WIDTH_P2; gj++) begin : gen_leaf
            assign nz[0][gj]  = (gj < WIDTH) ? xq[gj] : 1'b0;
            assign idx[0][gj] = '0;
        end

        for (gl = 1; gl <= NUM_LEVELS; gl++) begin : gen_level
            localparam int GROUPS = WIDTH_P2 >> gl;
            for (gj = 0; gj < GROUPS; gj++) begin : gen_group
                assign nz[gl][gj]  = nz[gl-1][2*gj+1] | nz[gl-1][2*gj];
                assign idx[gl][gj] = nz[gl-1][2*gj+1]
                                      ? (idx[gl-1][2*gj+1] | (MSB_W'(1) << (gl-1)))
                                      : idx[gl-1][2*gj];
            end
        end
    endgenerate

    logic [MSB_W-1:0] msb_tree;
    assign msb_tree = idx[NUM_LEVELS][0];

    logic [MSB_W-1:0]         msb;
    logic signed [AQ_W-1:0]   aq_full;
    logic [WIDTH-1:0]         normalized_x;
    logic [BITS-1:0]          fk;
    logic [LG_SEG:0]          seg;      // spare bit keeps SEGMENTS=1 (LG_SEG=0) a legal range
    logic signed [31:0]       yq_full;

    always_comb begin
        // Defaults so every variable is assigned on all paths (no inferred latches).
        msb          = '0;
        aq_full      = '0;
        normalized_x = '0;
        fk           = '0;
        seg          = '0;
        yq_full      = '0;

        if (xq != 0) begin
            msb          = msb_tree;                                   // floor(log2 xq), balanced tree above
            aq_full      = $signed({1'b0, msb}) - $signed(AQ_W'(BITS));
            normalized_x = xq << (WIDTH - 1 - msb);                    // leading one at bit WIDTH-1
            fk           = normalized_x[WIDTH-2 : WIDTH-1-BITS];       // mantissa fraction
            seg          = (LG_SEG == 0) ? '0 : (LG_SEG+1)'(fk >> (BITS - LG_SEG));
            yq_full      = ( (32'(aq_full) * coef_l(COEF_BITS)) <<< BITS )
                         + ( coef_a(SEGMENTS, COEF_BITS, int'(seg)) * $signed({1'b0, fk}) )
                         + ( coef_b(SEGMENTS, COEF_BITS, int'(seg)) <<< BITS )
                         + (1 <<< (COEF_BITS-1));
            yq_full      = yq_full >>> COEF_BITS;
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