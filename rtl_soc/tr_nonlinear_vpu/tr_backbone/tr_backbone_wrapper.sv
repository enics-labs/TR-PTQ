/*
 * @module   tr_backbone_wrapper
 * @brief    Encapsulated Log-Domain Math Pipeline (Vectorized).
 * @details  Chains TR-ln, post-LN modification, and TR-exp. Only lane 0 has the ln unit (softmax SM_P3 and RMSNorm RM_P2 read
 *           lane 0 only; every other use is bypass_ln), and post_ln only supports bypass and the rounded -0.5x (mode 11) on lane 0.
 *           Purely combinational. Expects wide inputs and natively steps down 
 *           precision through the tr_ln_alu block.
 *
 *           Pipeline (3 register stages, 3 cycles in_valid -> out_valid): input regs -> tr_ln_alu -> regs ->
 *           round-back/saturate + post_ln -> regs -> tr_exp_alu (combinational to the outputs).
 *
 *           The ONE shared tr_ln_alu runs at BITS=FRAC_W+LN_EXTRA (Q.6) so every operator uses the
 *           same unit: operators with FRAC_W fractional bits feed x<<LN_EXTRA (ln_pre_scaled=0),
 *           RMSNorm feeds its mean already at FRAC_W+LN_EXTRA bits (ln_pre_scaled=1). The ln result
 *           (Q.6, WIDTH_OUT+LN_EXTRA-ish bits) is rounded back to Q.FRAC_W and saturated to WIDTH_OUT.
 *
 * @param    N           Vector dimension.
 * @param    WIDTH_IN    Incoming datapath width (e.g., 16-bit).
 * @param    WIDTH_OUT   Target operating width for EXP/Mantissa (e.g., 8-bit).
 * @param    FRAC_W      Fractional bit width.
 * @param    LUT_IDX_W   Width of the ROM index for the Taylor anchor.
 * @param    LN_EXTRA    Extra fractional bits of the shared ln unit (BITS = FRAC_W + LN_EXTRA).
 * @param    LN_IN_W     Width of the shared ln unit's input. The largest reachable value is 4096 (RMSNorm mean, all codes -128):
 *                       softmax's slice x4 <= 512, so 13 bits suffice; anything larger is clamped to
 *                       2^LN_IN_W-1 (unreachable with the shipped controller).
 */
module tr_backbone_wrapper #(
    parameter int N         = 8,
    parameter int WIDTH_IN  = 16,
    parameter int WIDTH_OUT = 8,
    parameter int FRAC_W    = 4,
    parameter int LUT_IDX_W = 3,
    parameter int LN_EXTRA  = 2,
    parameter int LN_IN_W   = 13   // width of the ln unit's input; every reachable input fits (max 4096), larger values are clamped
)(
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    output logic out_valid,

    // Control Flags
    input  logic bypass_ln,
    input  logic [1:0]              mode_post_ln, // 00: bypass, 11: -0.5x (rounded, lane 0)
    input  logic ln_pre_scaled,  // 0: vec_in has FRAC_W frac bits (shifted up by LN_EXTRA), 1: already FRAC_W+LN_EXTRA
    
    // Datapath Input
    input  logic signed [WIDTH_IN-1:0]  vec_in [N],
    
    // Datapath Outputs
    output logic signed [WIDTH_OUT-1:0]   log_out [N],
    output logic [LUT_IDX_W-1:0]   a_idx_out [N],
    output logic [WIDTH_OUT-1:0]   mantisa_out [N],
    output logic is_zero_out [N]
);

    // =========================================================
    // STAGE 1: INPUT REGISTER (Fixes in2reg)
    // =========================================================
    logic signed [WIDTH_IN-1:0] s1_vec_in [N];
    logic                       s1_bypass_ln;
    logic [1:0]                 s1_mode_post_ln;
    logic                       s1_ln_pre_scaled;
    logic                       s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_bypass_ln <= 1'b0;
            s1_mode_post_ln <= '0;
            s1_ln_pre_scaled <= 1'b0;
            for(int i=0; i<N; i++) s1_vec_in[i] <= '0;
        end else begin
            s1_valid <= in_valid;
            s1_bypass_ln <= bypass_ln;
            s1_mode_post_ln <= mode_post_ln;
            s1_ln_pre_scaled <= ln_pre_scaled;
            s1_vec_in <= vec_in;
        end
    end

    // =========================================================
    // STAGE 1: TR-LN (Combinational) -> PIPELINE REGISTER 1b
    // =========================================================
    // The ln result is registered here, BEFORE the round-back / saturate / post-ln logic, to keep the stage within the
    // clock period (synthesis: tr_ln_alu ends at ~2.9 ns of the 3.3 ns budget; round-back + saturate + post_ln
    // add another ~0.8 ns). The bypass path (bypass_ln) is merged into the same register as the saturated 8-bit value
    // shifted up by LN_EXTRA, so the stage-2 round-back returns it unchanged ((4b+2)>>>2 = b).
    logic signed [WIDTH_OUT-1:0] tr_ln_out   [N];
    logic signed [WIDTH_OUT-1:0] post_ln_out [N];

    localparam int LN_OUT_W = FRAC_W + LN_EXTRA + 4;   // ln unit output width (Q.(FRAC_W+LN_EXTRA), 4 integer bits)
    localparam int MAX_VAL =  (1 << (WIDTH_OUT - 1)) - 1;   // e.g., for 8-bit: Max = 127, Min = -128
    localparam int MIN_VAL = -(1 << (WIDTH_OUT - 1));

    logic signed [LN_OUT_W-1:0] s1_ln_y [N];            // stage-1 result per lane: ln output, or the bypassed value << LN_EXTRA

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : gen_tr_lanes
            
            // Shared TR-LN ALU, BITS = FRAC_W + LN_EXTRA. Input is zero-extended (the unit treats it as unsigned).
            logic [WIDTH_IN+LN_EXTRA-1:0]      ln_xin_full;
            logic [LN_IN_W-1:0]                ln_xin;
            logic signed [LN_OUT_W-1:0]        ln_y;
            logic signed [WIDTH_OUT-1:0]       byp_sat;

            assign ln_xin_full = s1_ln_pre_scaled ? {{LN_EXTRA{1'b0}}, s1_vec_in[i]}
                                                  : {s1_vec_in[i], {LN_EXTRA{1'b0}}};
            assign ln_xin = (ln_xin_full >> LN_IN_W) != 0 ? {LN_IN_W{1'b1}} : ln_xin_full[LN_IN_W-1:0];

            if (i == 0) begin : gen_ln
                tr_ln_alu #(
                    .WIDTH(LN_IN_W),
                    .BITS(FRAC_W + LN_EXTRA),
                    .OUT_WIDTH(LN_OUT_W)
                ) u_tr_ln (
                    .xq (ln_xin),
                    .yq (ln_y)
                );
            end else begin : gen_no_ln
                assign ln_y = '0;                  // lanes 1..N-1 only ever bypass the ln
            end

            // Saturation clamp for the bypassed WIDTH_IN -> WIDTH_OUT cast
            always_comb begin
                if (s1_vec_in[i] > MAX_VAL)       byp_sat = WIDTH_OUT'(MAX_VAL);
                else if (s1_vec_in[i] < MIN_VAL)  byp_sat = WIDTH_OUT'(MIN_VAL);
                else                               byp_sat = WIDTH_OUT'(s1_vec_in[i]);
            end

            assign s1_ln_y[i] = (s1_bypass_ln || i != 0) ? (LN_OUT_W'(byp_sat) <<< LN_EXTRA) : ln_y;

        end
    endgenerate

    // --- PIPELINE REGISTER 1b ---
    logic signed [LN_OUT_W-1:0] s1b_ln_y [N];
    logic [1:0]                 s1b_mode_post_ln;
    logic                       s1b_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1b_valid <= 1'b0;
            s1b_mode_post_ln <= '0;
            for(int j=0; j<N; j++) s1b_ln_y[j] <= '0;
        end else begin
            s1b_valid <= s1_valid;
            s1b_mode_post_ln <= s1_mode_post_ln;
            s1b_ln_y <= s1_ln_y;
        end
    end

    // =========================================================
    // STAGE 2: ROUND-BACK / SATURATE & POST-LN (Combinational)
    // =========================================================
    logic signed [WIDTH_OUT-1:0] ln_mux_out [N];

    always_comb begin
        for (int i = 0; i < N; i++) begin
            logic signed [LN_OUT_W:0] rnd;
            logic signed [LN_OUT_W:0] r;
            // Round back to FRAC_W fractional bits, saturate to WIDTH_OUT.
            rnd = {s1b_ln_y[i][LN_OUT_W-1], s1b_ln_y[i]} + (1 <<< (LN_EXTRA-1));
            r   = rnd >>> LN_EXTRA;
            if (r > MAX_VAL)       ln_mux_out[i] = WIDTH_OUT'(MAX_VAL);
            else if (r < MIN_VAL)  ln_mux_out[i] = WIDTH_OUT'(MIN_VAL);
            else                   ln_mux_out[i] = WIDTH_OUT'(r);
        end
    end
    
    // Lanes 1..N-1 are never read as a log (bypass only); lane 0 supports bypass (00) and RMSNorm's rounded -0.5x (11):
    // -((x+1)>>>1), computed one bit wider so x=MAX does not wrap.
    logic signed [WIDTH_OUT:0] half_x;
    assign half_x = ($signed({ln_mux_out[0][WIDTH_OUT-1], ln_mux_out[0]}) + 1) >>> 1;
    always_comb begin
        for (int i = 0; i < N; i++) post_ln_out[i] = ln_mux_out[i];
        if (s1b_mode_post_ln == 2'b11) post_ln_out[0] = WIDTH_OUT'(-half_x);
    end

    // --- PIPELINE REGISTER 2 ---
    logic signed [WIDTH_OUT-1:0] s2_post_ln_reg [N];
    logic                        s2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            for(int j=0; j<N; j++) s2_post_ln_reg[j] <= '0;
        end else begin
            s2_valid <= s1b_valid;
            s2_post_ln_reg <= post_ln_out;
        end
    end

    // =========================================================
    // STAGE 3: TR-EXP (Combinational to Output)
    // =========================================================
    assign out_valid = s2_valid;
    assign log_out = s2_post_ln_reg;

    generate
        for (i = 0; i < N; i++) begin : gen_tr_exp_lanes
            
            // 4. TR-EXP ALU
            tr_exp_alu #(
                .WIDTH(WIDTH_OUT),
                .FRAC_W(FRAC_W),
                .LUT_IDX_W(LUT_IDX_W),
                .ITER(2) // Quadratic
            ) u_tr_exp (
                .x       (s2_post_ln_reg[i]),
                .a_idx   (a_idx_out[i]),
                .mantisa (mantisa_out[i]),
                .is_zero (is_zero_out[i])
            );
        end
    endgenerate

endmodule