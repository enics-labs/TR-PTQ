`timescale 1ns/1ps

/*
 * @module   ibert_rmsnorm
 * @brief    Standalone divider-based RMSNorm: I-BERT Newton integer square root plus one integer division per element, at the output
 *           precision of tr_rmsnorm (12-bit Q5.7 codes). Area / energy reference for the reciprocal-sqrt unit.
 * @details  Same command interface and sum-of-squares front end as tr_rmsnorm (rmsnorm_sumsq), so the two differ only in how 1/rms is formed
 *           and applied. Golden: tr_math_model.hpp ibert_rmsnorm_scale / ibert_rmsnorm_apply_tile.
 *             cmd 0/1  ACC_FIRST / ACC   S (+)= sum x^2 of the tile
 *             cmd 2    SCALE             msq = round(S * 4^F / N);  R = isqrt(msq) by Newton iteration (I-BERT Algorithm 4),
 *                                        each x <- (x + msq / x) >> 1 through the shared divider, until x stops decreasing
 *             cmd 3    APPLY             per lane (sequential, one divider): y = sign * min(round(|x| * 2^(7+F) / R), 2047 | 2048)
 *           R = rms code * 2^F (F = 5), reported on scale_o. The divider is the conventional restoring divider of the ISCAS I-BERT
 *           baselines (ibert_divider.sv), WIDTH cycles per division; the structure is one iterative divider with sequential lanes. All-zero input gives R = 0 and zero output.
 */
module ibert_rmsnorm #(
    parameter int N     = 8,
    parameter int W     = 8,
    parameter int OUT_W = 12,
    parameter int ACC_W = 32,
    parameter int F     = 5,
    parameter int OUT_FRAC = 7
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  in_valid,
    input  logic [1:0]            cmd,
    input  logic [3:0]            log2n,
    input  logic signed [W-1:0]   x_in [N],
    output logic                  out_valid,
    output logic signed [OUT_W-1:0] y_out [N],
    output logic [15:0]           scale_o
);

    localparam logic [1:0] CMD_SCALE = 2'd2, CMD_APPLY = 2'd3;
    localparam int MSQ_W = 2 * (W - 1) + 2 * F + 1;                  // mean of squares with 2F fractional bits (<= 2^14 * 4^F)
    localparam int DIV_W = MSQ_W;
    localparam int Q_SH  = OUT_FRAC + F;                             // dividend shift of the apply division
    localparam int MAXQ_POS = (1 << (OUT_W - 1)) - 1;
    localparam int MAXQ_NEG = (1 << (OUT_W - 1));

    typedef enum logic [3:0] {
        IDLE, ACC_WAIT, SC_MSQ, SQRT_INIT, SQRT_ISSUE, SQRT_WAIT, SQRT_STEP, SC_DONE,
        AP_ISSUE, AP_WAIT, AP_DONE
    } state_t;
    state_t state;

    // ---------------- accumulate ----------------
    logic [ACC_W-1:0] S;
    logic             ss_done;
    rmsnorm_sumsq #(.N(N), .W(W), .ACC_W(ACC_W)) u_ss (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid && !cmd[1]), .in_first(cmd == 2'd0),
        .x_in(x_in), .done(ss_done), .S(S)
    );

    // ---------------- shared restoring divider ----------------
    logic               div_valid_in, div_valid_out;
    logic [DIV_W-1:0]   div_dividend, div_divisor, div_quotient;
    ibert_divider #(.WIDTH(DIV_W)) u_div (
        .clk(clk), .rst_n(rst_n), .valid_in(div_valid_in), .dividend(div_dividend), .divisor(div_divisor),
        .busy(), .valid_out(div_valid_out), .quotient(div_quotient), .remainder()
    );

    logic [MSQ_W-1:0]   msq, guess, R;
    logic [4:0]         lane;
    logic signed [W-1:0] xr [N];
    logic signed [OUT_W-1:0] yr [N];

    // |x| of the lane being divided
    logic [W-1:0] ax;
    always_comb ax = xr[lane][W-1] ? W'(-(W+1)'(xr[lane])) : W'(xr[lane]);

    // div_valid_in is a registered pulse raised in the *_ISSUE state, so the operands are sampled during the *_WAIT state: the mux must be
    // stable over both states of a phase .
    always_comb begin
        if (state == SQRT_ISSUE || state == SQRT_WAIT) begin
            div_dividend = msq;
            div_divisor  = guess;
        end else begin
            div_dividend = DIV_W'(({{(DIV_W-W){1'b0}}, ax} << Q_SH) + (R >> 1));
            div_divisor  = (R == '0) ? DIV_W'(1) : DIV_W'(R);
        end
    end

    // leading-one position of msq and the Newton starting point 2^ceil(bits/2)
    logic [$clog2(MSQ_W+1)-1:0] bits_n;
    always_comb begin
        bits_n = '0;
        for (int k = 0; k < MSQ_W; k++) if (msq[k]) bits_n = ($clog2(MSQ_W+1))'(k + 1);
    end

    logic [MSQ_W:0] x_next;
    always_comb x_next = ({1'b0, guess} + {1'b0, div_quotient[MSQ_W-1:0]}) >> 1;

    // rounded mean of squares: (S << 2F) + (N / 2)) >> log2n
    logic [ACC_W+2*F:0] msq_wide;
    always_comb msq_wide = (({1'b0, S} << (2 * F)) + ((ACC_W+2*F+1)'(1) << (log2n - 1))) >> log2n;

    logic [DIV_W-1:0] q_lim;
    always_comb q_lim = div_quotient;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; out_valid <= 1'b0; div_valid_in <= 1'b0;
            msq <= '0; guess <= '0; R <= '0; lane <= '0;
            for (int i = 0; i < N; i++) begin xr[i] <= '0; yr[i] <= '0; y_out[i] <= '0; end
        end else begin
            out_valid    <= 1'b0;
            div_valid_in <= 1'b0;
            case (state)
                IDLE: begin
                    if (in_valid) begin
                        if (!cmd[1])             state <= ACC_WAIT;
                        else if (cmd == CMD_SCALE) state <= SC_MSQ;
                        else begin xr <= x_in; lane <= '0; state <= AP_ISSUE; end
                    end
                end
                ACC_WAIT: if (ss_done) begin out_valid <= 1'b1; state <= IDLE; end
                SC_MSQ: begin
                    msq   <= msq_wide[MSQ_W-1:0];
                    state <= SQRT_INIT;
                end
                SQRT_INIT: begin
                    if (msq == '0) begin
                        R     <= '0;
                        state <= SC_DONE;
                    end else begin
                        guess <= MSQ_W'(1) << ((bits_n + 1) >> 1);
                        state <= SQRT_ISSUE;
                    end
                end
                SQRT_ISSUE: begin div_valid_in <= 1'b1; state <= SQRT_WAIT; end
                SQRT_WAIT:  if (div_valid_out) state <= SQRT_STEP;
                SQRT_STEP: begin
                    if (x_next >= {1'b0, guess}) begin
                        R     <= guess;
                        state <= SC_DONE;
                    end else begin
                        guess <= x_next[MSQ_W-1:0];
                        state <= SQRT_ISSUE;
                    end
                end
                SC_DONE: begin out_valid <= 1'b1; state <= IDLE; end
                AP_ISSUE: begin div_valid_in <= 1'b1; state <= AP_WAIT; end
                AP_WAIT: begin
                    if (div_valid_out) begin
                        if (xr[lane][W-1])
                            yr[lane] <= -OUT_W'((q_lim > DIV_W'(MAXQ_NEG)) ? DIV_W'(MAXQ_NEG) : q_lim);
                        else
                            yr[lane] <=  OUT_W'((q_lim > DIV_W'(MAXQ_POS)) ? DIV_W'(MAXQ_POS) : q_lim);
                        if (lane == N - 1) state <= AP_DONE;
                        else begin lane <= lane + 1'b1; state <= AP_ISSUE; end
                    end
                end
                AP_DONE: begin y_out <= yr; out_valid <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    assign scale_o = 16'(R);

endmodule
