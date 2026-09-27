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
    end

    localparam int QLN2 = 11;   // round(ln(2) * 16)
    localparam int QB   = 21;   // floor(1.353 * 16)
    localparam int QC   = 245;  // floor(0.344 * 256 / 0.3585)
    localparam int ACC_W = W + 8 + $clog2(N);  // headroom for exp_code sum

    typedef enum logic [2:0] {IDLE, EXP, DIV_ISSUE, DIV_WAIT, DONE} state_t;
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
    // Combinational max + i-Exp evaluation (Stage EXP inputs are x_reg)
    // ---------------------------------------------------------------
    logic signed [W-1:0]  max_x;
    logic [ACC_W-1:0]      exp_comb [N];
    logic [ACC_W-1:0]      sum_comb;

    always_comb begin
        max_x = x_reg[0];
        for (int i = 1; i < N; i++)
            if (x_reg[i] > max_x) max_x = x_reg[i];

        sum_comb = '0;
        for (int i = 0; i < N; i++) begin
            logic signed [31:0] diff, z, qp, delta, delta_sq, qout, shifted;
            diff = 32'(x_reg[i]) - 32'(max_x);       // <= 0
            z    = (-diff) / QLN2;                    // divide by compile-time constant
            qp   = diff + z * QLN2;
            delta    = qp + QB;
            delta_sq = delta * delta;
            qout     = delta_sq + QC;                  // always > 0
            shifted  = qout >>> z;
            exp_comb[i] = ACC_W'(shifted);
            sum_comb = sum_comb + ACC_W'(shifted);
        end
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
            for (int i = 0; i < N; i++) begin
                x_reg[i] <= '0; exp_code[i] <= '0; y_reg[i] <= '0; y_out[i] <= '0;
            end
        end else begin
            valid_out    <= 1'b0;
            div_valid_in <= 1'b0;

            case (state)
                IDLE: begin
                    if (valid_in) begin
                        x_reg <= x_in;
                        state <= EXP;
                    end
                end

                EXP: begin
                    exp_code <= exp_comb;
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
            endcase
        end
    end

endmodule
