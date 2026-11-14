`timescale 1ns/1ps

/*
 * @module   rmsnorm_sumsq
 * @brief    Sum-of-squares front end of the standalone RMSNorm modules: S += sum(x[i]^2) over one N-lane tile.
 * @details  Shared by tr_rmsnorm and ibert_rmsnorm so that both pay for the same accumulate stage. Four pipeline stages:
 *             1 input registers, 2 per-lane squares (15-bit), 3 balanced adder tree (log2 N levels), 4 accumulate into S.
 *           in_first starts a new row (S = tile sum), otherwise S += tile sum. done pulses one cycle after S is updated.
 *           Bit-exact to tr_math_model.hpp::rms_acc_tile.
 */
module rmsnorm_sumsq #(
    parameter int N     = 8,
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 in_valid,
    input  logic                 in_first,
    input  logic signed [W-1:0]  x_in [N],
    output logic                 done,
    output logic [ACC_W-1:0]     S
);

    localparam int SQ_W  = 2 * W - 1;                 // |x|^2 <= 2^(2W-2)
    localparam int SUM_W = SQ_W + $clog2(N);

    logic                 v1, f1;
    logic signed [W-1:0]  x1 [N];
    logic                 v2, f2;
    logic [SQ_W-1:0]      sq2 [N];
    logic                 v3, f3;
    logic [SUM_W-1:0]     sum3;

    logic signed [2*W-1:0] p [N];
    always_comb for (int i = 0; i < N; i++) p[i] = x1[i] * x1[i];

    logic [SUM_W-1:0] lvl [N];
    always_comb begin
        logic [SUM_W-1:0] t [N];
        for (int i = 0; i < N; i++) t[i] = SUM_W'(sq2[i]);
        for (int lv = 0; lv < $clog2(N); lv++)
            for (int i = 0; i < (N >> (lv + 1)); i++) t[i] = t[2*i] + t[2*i+1];
        for (int i = 0; i < N; i++) lvl[i] = t[i];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0; f1 <= 1'b0; v2 <= 1'b0; f2 <= 1'b0; v3 <= 1'b0; f3 <= 1'b0;
            sum3 <= '0; S <= '0; done <= 1'b0;
            for (int i = 0; i < N; i++) begin x1[i] <= '0; sq2[i] <= '0; end
        end else begin
            v1 <= in_valid; f1 <= in_first;
            if (in_valid) x1 <= x_in;
            v2 <= v1; f2 <= f1;
            if (v1) for (int i = 0; i < N; i++) sq2[i] <= p[i][SQ_W-1:0];
            v3 <= v2; f3 <= f2;
            if (v2) sum3 <= lvl[0];
            done <= v3;
            if (v3) S <= f3 ? ACC_W'(sum3) : S + ACC_W'(sum3);
        end
    end

endmodule
