`timescale 1ns/1ps

/*
 * @module   tr_softmax
 * @brief    Standalone softmax tile with the Taylor-Region exponential and logarithm (no divider), without the shared VPU backbone.
 * @details  The arithmetic of the production flow (golden: tr_math_model.hpp softmax_hw_model_ex), built as a dedicated block so that its
 *           area can be compared with the divider-based ibert_softmax and with the share of the VPU that softmax uses. One command per
 *           8-lane tile (in_valid with x_in), the four passes of the controller run internally on one set of 8 exponential lanes:
 *             max   = max over the lanes
 *             pass 1  d_i = sat8(x_i - max);  S = sum_i exp(d_i)            exp = tr_exp_alu + 8-entry LUT + product (is_zero -> 256)
 *             ln    ln(S) from S[23:8] through the shared-format tr_ln_alu (BITS = 6, input x4, rounded back to Q4.4, saturated)
 *             z     = sat8(max + ln(S))
 *             pass 2  p_i = min(255, exp(sat8(x_i - z)) >> 4), Q0.8 unsigned
 *           out_valid pulses when p_out (8 bytes) and z_o (the tile's log-partition value, int8) are valid.
 */
module tr_softmax #(
    parameter int N = 8,
    parameter int W = 8
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 in_valid,
    input  logic signed [W-1:0]  x_in [N],
    output logic                 out_valid,
    output logic [W-1:0]         p_out [N],
    output logic signed [W-1:0]  z_o
);

    localparam int FRAC_W   = 4;
    localparam int LN_EXTRA = 2;
    localparam int LN_IN_W  = 13;
    localparam int LN_OUT_W = FRAC_W + LN_EXTRA + 4;
    localparam int PROD_W   = 16;
    localparam int S_W      = PROD_W + $clog2(N);

    typedef enum logic [3:0] {IDLE, MAXF, SUB1, EXP1, SUM, LN, ZSUB, SUB2, EXP2, DONE} state_t;
    state_t state;

    logic signed [W-1:0] x_r [N];
    logic signed [W-1:0] max_r, z_r;
    logic signed [W-1:0] d_r [N];
    logic [PROD_W-1:0]   prod_r [N];
    logic [S_W-1:0]      S_r;
    logic signed [LN_OUT_W-1:0] ln_r;

    // ---------------- max ----------------
    logic signed [W-1:0] max_c;
    always_comb begin
        logic signed [W-1:0] t [N];
        for (int i = 0; i < N; i++) t[i] = x_r[i];
        for (int lv = 0; lv < $clog2(N); lv++)
            for (int i = 0; i < (N >> (lv + 1)); i++) t[i] = (t[2*i] > t[2*i+1]) ? t[2*i] : t[2*i+1];
        max_c = t[0];
    end

    // ---------------- saturating subtract (pass 1: max, pass 2: z) ----------------
    logic signed [W-1:0]   sub_b;
    logic signed [W-1:0]   d_c [N];
    assign sub_b = (state == SUB2) ? z_r : max_r;
    always_comb
        for (int i = 0; i < N; i++) begin
            logic signed [W:0] diff;
            diff   = (W+1)'(x_r[i]) - (W+1)'(sub_b);
            d_c[i] = (diff > 127) ? W'(127) : (diff < -128) ? W'(-128) : W'(diff[W-1:0]);
        end

    // ---------------- exponential lanes (shared by both passes) ----------------
    logic [2:0]          a_idx [N];
    logic [W-1:0]        mant  [N];
    logic                is_zero [N];
    logic [7:0]          e_a [N];
    logic [PROD_W-1:0]   prod_c [N];
    genvar gi;
    generate
        for (gi = 0; gi < N; gi++) begin : gen_exp
            tr_exp_alu #(.WIDTH(W), .FRAC_W(FRAC_W), .LUT_IDX_W(3), .ITER(2)) u_exp (
                .x(d_r[gi]), .a_idx(a_idx[gi]), .mantisa(mant[gi]), .is_zero(is_zero[gi])
            );
            logic [PROD_W:0] lut_v;
            assign lut_v      = is_zero[gi] ? (PROD_W+1)'(256) : (PROD_W+1)'(e_a[gi]);    // is_zero: 256 (anchor doubling trick)
            assign prod_c[gi] = PROD_W'(lut_v * (PROD_W+1)'(mant[gi]));
        end
    endgenerate
    shared_lut_rom #(.N(N), .LUT_IDX_W(3)) u_lut (.a_idx(a_idx), .e_a(e_a));

    // ---------------- sum of the exponentials ----------------
    logic [S_W-1:0] sum_c;
    always_comb begin
        logic [S_W-1:0] t [N];
        for (int i = 0; i < N; i++) t[i] = S_W'(prod_r[i]);
        for (int lv = 0; lv < $clog2(N); lv++)
            for (int i = 0; i < (N >> (lv + 1)); i++) t[i] = t[2*i] + t[2*i+1];
        sum_c = t[0];
    end

    // ---------------- ln(S) through the ln unit of the shared flow ----------------
    logic [LN_IN_W-1:0]         ln_xin;
    logic signed [LN_OUT_W-1:0] ln_y;
    logic [FRAC_W+LN_EXTRA-1:0] unused_fk;
    logic [$clog2(LN_IN_W)-1:0] unused_msb;
    always_comb begin
        logic [15:0] s16;
        logic [15+LN_EXTRA:0] s16x;
        s16    = 16'((S_r >> 8) & 16'hFFFF);
        s16x   = {s16, {LN_EXTRA{1'b0}}};
        ln_xin = (s16x >> LN_IN_W) != 0 ? {LN_IN_W{1'b1}} : s16x[LN_IN_W-1:0];
    end
    tr_ln_alu #(.WIDTH(LN_IN_W), .BITS(FRAC_W + LN_EXTRA), .OUT_WIDTH(LN_OUT_W)) u_ln (
        .xq(ln_xin), .yq(ln_y), .msb_o(unused_msb), .fk_o(unused_fk)
    );

    // round back to Q4.4 and saturate, z = sat8(max + ln(S))
    logic signed [W-1:0] z_c;
    always_comb begin
        logic signed [LN_OUT_W:0] rnd, r;
        logic signed [W:0]        zz;
        rnd = {ln_r[LN_OUT_W-1], ln_r} + (1 <<< (LN_EXTRA - 1));
        r   = rnd >>> LN_EXTRA;
        zz  = (W+1)'(max_r) + (W+1)'((r > 127) ? 127 : (r < -128) ? -128 : r);
        z_c = (zz > 127) ? W'(127) : (zz < -128) ? W'(-128) : W'(zz[W-1:0]);
    end

    // ---------------- probabilities: exp >> 4, saturate to 255 ----------------
    logic [W-1:0] p_c [N];
    always_comb
        for (int i = 0; i < N; i++) p_c[i] = ((prod_r[i] >> 4) > 255) ? 8'd255 : W'(prod_r[i] >> 4);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; out_valid <= 1'b0; max_r <= '0; z_r <= '0; S_r <= '0; ln_r <= '0;
            for (int i = 0; i < N; i++) begin x_r[i] <= '0; d_r[i] <= '0; prod_r[i] <= '0; p_out[i] <= '0; end
        end else begin
            out_valid <= 1'b0;
            case (state)
                IDLE: if (in_valid) begin x_r <= x_in; state <= MAXF; end
                MAXF: begin max_r <= max_c; state <= SUB1; end
                SUB1: begin d_r <= d_c; state <= EXP1; end
                EXP1: begin prod_r <= prod_c; state <= SUM; end
                SUM:  begin S_r <= sum_c; state <= LN; end
                LN:   begin ln_r <= ln_y; state <= ZSUB; end
                ZSUB: begin z_r <= z_c; state <= SUB2; end
                SUB2: begin d_r <= d_c; state <= EXP2; end
                EXP2: begin prod_r <= prod_c; state <= DONE; end
                DONE: begin p_out <= p_c; out_valid <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    assign z_o = z_r;

endmodule
