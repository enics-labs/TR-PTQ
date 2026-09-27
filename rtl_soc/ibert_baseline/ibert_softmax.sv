`timescale 1ns/1ps

/*
 * @module   ibert_softmax
 * @brief    I-BERT-style integer-only Softmax (Kim et al., "I-BERT:
 *           Integer-only BERT Quantization", ICML 2021 -- Eq. 10-14,
 *           Algorithm 3).
 * @details  max-subtract -> i-Exp (second-order polynomial exponential,
 *           exp(p)>>>z decomposition, p=x-max reduced mod ln2) -> integer
 *           division for the normalization sum. This is a genuinely
 *           standalone, self-contained unit: no shared exp/ln backbone with
 *           GELU/RMSNorm, and (unlike the shared TR-VPU, whose softmax avoids
 *           any divider via a log-sum-exp trick) this baseline needs an
 *           explicit runtime division per lane.
 *
 *           I-BERT's own Algorithm 3 does not specify a divider design for
 *           "qexp/sum(qexp)" -- this uses ibert_divider.sv (a conventional
 *           restoring divider), a design choice made for this baseline, not
 *           part of I-BERT itself. It is instantiated once and reused
 *           sequentially across the 8 per-lane normalization divisions.
 *
 *           Fixed-point realization, Q4.4 native I/O: the polynomial stays in
 *           the SAME Q4.4 domain as x throughout (Algorithm 3 reuses the same
 *           scale S for both x and p, unlike GELU's erf substitution), so no
 *           extra internal precision is needed here. qln2=round(ln2*16)=11 is
 *           a compile-time constant divisor -- dividing by it is done
 *           directly with SystemVerilog's '/' operator (a divide-by-constant
 *           reduces to shifts/adds and is not the same hardware cost as the
 *           runtime sum-divide, so it is not routed through ibert_divider).
 *           Softmax output is Q0.8 UNSIGNED (0..255, saturating), matching
 *           the TR-VPU comparison point's own softmax convention. Only Q4.4
 *           (N=8, W=8, FRAC_W=4) is implemented/verified.
 *
 *           Timing fix: the original single-state EXP computation used
 *           generic 32-bit signed intermediates for every per-lane term
 *           (diff/z/qp/delta/delta_sq/qout/shifted), plus an 8-way *linear*
 *           max-scan and an 8-way *linear* sum-accumulate -- both are
 *           sequentially-dependent chains of depth N=8, the same class of
 *           bug that made ibert_gelu miss its 3.3ns reg2reg target. This
 *           version hand-sizes every intermediate to its actual verified
 *           numeric range (see comments below), replaces the linear max-scan
 *           and sum-accumulate with balanced log-depth reduction trees
 *           (depth 3 for N=8), and splits the per-lane exponential chain
 *           across three states (MAXFIND / EXP1 / EXP2) so no single cycle
 *           chains "subtract -> divide-by-const -> multiply -> square ->
 *           variable-shift -> reduce" combinationally.
 *
 * @param    N       Number of parallel lanes.
 * @param    W       Data width (Q4.4 native: 8).
 * @param    FRAC_W  Fractional bits (Q4.4 native: 4).
 * @param    DIV_W   ibert_divider operand width.
 */
module ibert_softmax #(
    parameter int N      = 8,
    parameter int W      = 8,
    parameter int FRAC_W = 4,
    parameter int DIV_W  = 24
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [W-1:0] x_in [N],
    output logic valid_out,
    output logic [W-1:0] y_out [N]   // Q0.8 unsigned probability, 0..255
);

    initial begin
        if (W != 8 || FRAC_W != 4)
            $error("ibert_softmax: only W=8/FRAC_W=4 (Q4.4) is implemented/verified");
        if (N != 8)
            $error("ibert_softmax: the balanced max/sum reduction trees assume N=8");
    end

    localparam int QLN2 = 11;   // round(ln(2) * 16)
    localparam int QB   = 21;   // floor(1.353 * 16)
    localparam int QC   = 245;  // floor(0.344 * 256 / 0.3585)
    localparam int ACC_W = W + 8 + $clog2(N);  // headroom for exp_code sum

    typedef enum logic [3:0] {
        IDLE, MAXFIND, EXP1, EXP2, DIV_ISSUE, DIV_WAIT, DONE
    } state_t;
    state_t state;

    logic signed [W-1:0]   x_reg [N];
    logic [ACC_W-1:0]      exp_code [N];
    logic [ACC_W-1:0]      sum_reg;
    logic [W-1:0]           y_reg [N];
    logic [$clog2(N)-1:0]   lane_cnt;

    // ---------------------------------------------------------------
    // Divider instance, shared across the N per-lane normalization divides
    // ---------------------------------------------------------------
    logic                  div_valid_in;
    logic [DIV_W-1:0]      div_dividend, div_divisor;
    logic                  div_busy, div_valid_out;
    logic [DIV_W-1:0]      div_quotient;

    ibert_divider #(.WIDTH(DIV_W)) u_div (
        .clk(clk), .rst_n(rst_n),
        .valid_in(div_valid_in), .dividend(div_dividend), .divisor(div_divisor),
        .busy(div_busy), .valid_out(div_valid_out), .quotient(div_quotient), .remainder()
    );

    assign div_dividend = DIV_W'((exp_code[lane_cnt] << 8) + (sum_reg >> 1));  // rounding bias
    assign div_divisor  = DIV_W'(sum_reg);

    // ---------------------------------------------------------------
    // Stage MAXFIND: balanced 3-level max-reduction tree over x_reg (depth
    // log2(N)=3, replacing the original 8-way linear scan)
    // ---------------------------------------------------------------
    logic signed [W-1:0] max_l1 [N/2];
    logic signed [W-1:0] max_l2 [N/4];
    logic signed [W-1:0] max_comb;

    always_comb begin
        for (int i = 0; i < N/2; i++)
            max_l1[i] = (x_reg[2*i] > x_reg[2*i+1]) ? x_reg[2*i] : x_reg[2*i+1];
        for (int i = 0; i < N/4; i++)
            max_l2[i] = (max_l1[2*i] > max_l1[2*i+1]) ? max_l1[2*i] : max_l1[2*i+1];
        max_comb = (max_l2[0] > max_l2[1]) ? max_l2[0] : max_l2[1];
    end

    logic signed [W-1:0] max_x;

    // ---------------------------------------------------------------
    // Stage EXP1: diff = x-max (<=0), z = floor(-diff/QLN2), qp = diff+z*QLN2
    // (remainder of the division, always in (-QLN2,0]), delta = qp+QB.
    // Hand-verified ranges (x_reg,max_x in [-128,127]):
    //   diff  in [-255,0]      -> signed 9-bit
    //   negd  in [0,255]       -> unsigned 9-bit (= -diff)
    //   z     in [0,23]        -> unsigned 5-bit (255/11 floor = 23)
    //   qp    in [-10,0]       -> signed 5-bit
    //   delta in [11,21]       -> unsigned 6-bit
    // ---------------------------------------------------------------
    logic [5:0] delta1 [N];   // unsigned, range [11,21]
    logic [4:0] z1     [N];   // unsigned, range [0,23]

    always_comb begin
        for (int i = 0; i < N; i++) begin
            logic signed [8:0] diff;
            logic [8:0]        negd;
            logic [4:0]        z;
            logic [8:0]        zqln2_mag;   // unsigned magnitude of z*QLN2, in [0,253]
            logic signed [9:0] qp_wide;
            logic signed [4:0] qp;
            logic signed [5:0] delta_s;

            diff      = 9'(x_reg[i]) - 9'(max_x);
            negd      = 9'(-diff);
            z         = 5'(negd / 9'(QLN2));
            zqln2_mag = 9'(z) * 9'(QLN2);
            // zqln2_mag is always non-negative, so zero-extending it into a
            // signed container before adding keeps the whole sum in signed
            // arithmetic -- mixing a signed 'diff' with an unsigned operand
            // here would silently make the entire addition unsigned and
            // corrupt the (negative) result.
            qp_wide   = 10'(diff) + $signed({1'b0, zqln2_mag});
            qp        = 5'(qp_wide);

            z1[i]     = z;
            delta_s   = 6'(qp) + 6'(QB);
            delta1[i] = 6'(delta_s);   // delta_s in [11,21], always non-negative
        end
    end

    logic [5:0] delta_reg [N];
    logic [4:0] z_reg     [N];

    // ---------------------------------------------------------------
    // Stage EXP2: delta_sq = delta^2, qout = delta_sq+QC, shifted = qout>>>z,
    // then a balanced 3-level sum-reduction tree (replacing the original
    // 8-way linear accumulate).
    //   delta_sq in [121,441]  -> unsigned 9-bit
    //   qout     in [366,686]  -> unsigned 10-bit
    //   shifted  in [0,686]    -> unsigned 10-bit
    // ---------------------------------------------------------------
    logic [9:0] shifted2 [N];

    always_comb begin
        for (int i = 0; i < N; i++) begin
            logic [8:0]  delta_sq;
            logic [9:0]  qout;

            delta_sq     = 9'(delta_reg[i]) * 9'(delta_reg[i]);
            qout         = 10'(delta_sq) + 10'(QC);
            shifted2[i]  = qout >> z_reg[i];
        end
    end

    logic [10:0] sum_l1 [N/2];
    logic [11:0] sum_l2 [N/4];
    logic [ACC_W-1:0] sum_comb;

    always_comb begin
        for (int i = 0; i < N/2; i++)
            sum_l1[i] = 11'(shifted2[2*i]) + 11'(shifted2[2*i+1]);
        for (int i = 0; i < N/4; i++)
            sum_l2[i] = 12'(sum_l1[2*i]) + 12'(sum_l1[2*i+1]);
        sum_comb = ACC_W'(sum_l2[0]) + ACC_W'(sum_l2[1]);
    end

    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            valid_out <= 1'b0;
            div_valid_in <= 1'b0;
            lane_cnt <= '0;
            sum_reg <= '0;
            max_x <= '0;
            for (int i = 0; i < N; i++) begin
                x_reg[i] <= '0; exp_code[i] <= '0; y_reg[i] <= '0; y_out[i] <= '0;
                delta_reg[i] <= '0; z_reg[i] <= '0;
            end
        end else begin
            valid_out    <= 1'b0;
            div_valid_in <= 1'b0;

            case (state)
                IDLE: begin
                    if (valid_in) begin
                        x_reg <= x_in;
                        state <= MAXFIND;
                    end
                end

                MAXFIND: begin
                    max_x <= max_comb;
                    state <= EXP1;
                end

                EXP1: begin
                    delta_reg <= delta1;
                    z_reg     <= z1;
                    state     <= EXP2;
                end

                EXP2: begin
                    for (int i = 0; i < N; i++) exp_code[i] <= ACC_W'(shifted2[i]);
                    sum_reg  <= sum_comb;
                    lane_cnt <= '0;
                    state    <= DIV_ISSUE;
                end

                DIV_ISSUE: begin
                    div_valid_in <= 1'b1;
                    state <= DIV_WAIT;
                end

                DIV_WAIT: begin
                    if (div_valid_out) begin
                        y_reg[lane_cnt] <= (div_quotient > DIV_W'(255)) ? W'(255) : W'(div_quotient);
                        if (lane_cnt == N - 1) begin
                            state <= DONE;
                        end else begin
                            lane_cnt <= lane_cnt + 1'b1;
                            state    <= DIV_ISSUE;
                        end
                    end
                end

                DONE: begin
                    y_out     <= y_reg;
                    valid_out <= 1'b1;
                    state     <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
