`timescale 1ns/1ps

/*
 * @module   swiglu_core
 * @brief    Shared back end of the standalone SwiGLU modules: silu register, running max of |silu * up|, row gain, apply.
 * @details  Used by tr_swiglu and ibert_swiglu so that both pay for the same product / max / gain / apply stages; the two modules differ
 *           only in how the 12-bit silu of the gate group is produced. Golden: tr_math_model.hpp sw_abs_prod, sw2_gain, sw2_apply.
 *             silu_load        silu_r <= silu_in (the 8 silu codes of the current group, 12-bit signed)
 *             op 0  MAX        running max <= max(max, |silu_r * up|) over the 8 lanes                  (4 cycles to done)
 *             op 1  GAIN       e = largest k in [-8, 10] with (max << 2) << k <= 2047 << 10; clears the max (2 cycles to done)
 *             op 2  APPLY      y = sat12(round_shift((silu_r * up) << 2, 10 - e))                       (3 cycles to done)
 *           op_done pulses when the operation has completed; e_o is the row gain register.
 */
module swiglu_core #(
    parameter int N      = 8,
    parameter int W      = 8,
    parameter int SW_W   = 12,
    parameter int OUT_W  = 12
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    silu_load,
    input  logic signed [SW_W-1:0]  silu_in [N],
    input  logic                    op_valid,
    input  logic [1:0]              op,
    input  logic signed [W-1:0]     up_in [N],
    output logic                    op_done,
    output logic signed [OUT_W-1:0] y_out [N],
    output logic signed [4:0]       e_o
);

    localparam logic [1:0] OP_MAX = 2'd0, OP_GAIN = 2'd1, OP_APPLY = 2'd2;
    localparam int DROP    = 2;                       // silu_w (14 bit) -> silu_s (12 bit)
    localparam int TOTAL   = 10;                      // product fractional bits (6 + 4)
    localparam int PROD_W  = SW_W + W;                // 20
    localparam int MAXP_W  = PROD_W - 1;              // |prod| <= 2^18
    localparam int YMAX    = (1 << (OUT_W - 1)) - 1;
    localparam int YMIN    = -(1 << (OUT_W - 1));
    localparam int MX_W    = MAXP_W + DROP;           // max << 2

    logic signed [SW_W-1:0] silu_r [N];

    logic                   vA, vB, vC;
    logic [1:0]             opA, opB, opC;
    logic signed [W-1:0]    up_r [N];
    logic signed [PROD_W-1:0] prod_r [N];
    logic [MAXP_W-1:0]      mC, maxreg;

    // ---- |prod| and the max of the 8 lanes ----
    logic [MAXP_W-1:0] absp [N];
    logic [MAXP_W-1:0] tmax;
    always_comb begin
        logic [MAXP_W-1:0] t [N];
        for (int i = 0; i < N; i++) begin
            absp[i] = MAXP_W'(prod_r[i][PROD_W-1] ? -prod_r[i] : prod_r[i]);
            t[i]    = absp[i];
        end
        for (int lv = 0; lv < $clog2(N); lv++)
            for (int i = 0; i < (N >> (lv + 1)); i++) t[i] = (t[2*i] > t[2*i+1]) ? t[2*i] : t[2*i+1];
        tmax = t[0];
    end

    // ---- row gain from the running max ----
    logic [MX_W-1:0]        mx;
    logic signed [4:0]      e_c;
    always_comb begin
        mx = {maxreg, 2'b00};
        if (mx == '0) mx = MX_W'(1);
        e_c = -5'sd8;                                  // no threshold met: smallest gain (same as j = 18)
        for (int j = 18; j >= 0; j--)                  // the smallest j with mx <= 2047 << j wins (it is assigned last)
            if ({1'b0, mx} <= (32'(2047) << j)) e_c = 5'(10 - j);
    end

    // ---- apply: round_shift((silu * up) << 2, 10 - e), saturate ----
    logic signed [OUT_W-1:0] y_c [N];
    logic [4:0]              sh;
    always_comb begin
        sh = 5'(10 - e_o);                              // 0 .. 18
        for (int i = 0; i < N; i++) begin
            logic signed [PROD_W+DROP:0] p4, r, rnd;
            p4  = {prod_r[i][PROD_W-1], prod_r[i], 2'b00};
            rnd = (PROD_W+DROP+1)'(1) <<< (sh - 1);
            r   = (sh == 0) ? p4 : ((p4 + rnd) >>> sh);
            y_c[i] = (r > YMAX) ? OUT_W'(YMAX) : (r < YMIN) ? OUT_W'(YMIN) : OUT_W'(r);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vA <= 1'b0; vB <= 1'b0; vC <= 1'b0; opA <= '0; opB <= '0; opC <= '0;
            mC <= '0; maxreg <= '0; e_o <= 5'sd0; op_done <= 1'b0;
            for (int i = 0; i < N; i++) begin silu_r[i] <= '0; up_r[i] <= '0; prod_r[i] <= '0; y_out[i] <= '0; end
        end else begin
            if (silu_load) silu_r <= silu_in;

            vA <= op_valid; opA <= op;
            if (op_valid) up_r <= up_in;

            vB <= vA; opB <= opA;
            if (vA && opA != OP_GAIN)
                for (int i = 0; i < N; i++) prod_r[i] <= silu_r[i] * up_r[i];
            if (vA && opA == OP_GAIN) begin
                e_o    <= e_c;
                maxreg <= '0;
            end

            vC <= vB; opC <= opB;
            if (vB && opB == OP_MAX)   mC <= tmax;
            if (vB && opB == OP_APPLY) y_out <= y_c;

            if (vC && opC == OP_MAX && mC > maxreg) maxreg <= mC;

            op_done <= (vA && opA == OP_GAIN) || (vB && opB == OP_APPLY) || (vC && opC == OP_MAX);
        end
    end

endmodule
