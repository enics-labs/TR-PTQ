`timescale 1ns/1ps

/*
 * @module   tr_swiglu
 * @brief    Standalone SwiGLU with the symmetric 64-entry sigma table (12-bit silu and output codes), without the shared VPU backbone.
 * @details  The arithmetic of the production two-pass flow (golden: tr_math_model.hpp sw2_silu, sw_abs_prod, sw2_gain, sw2_apply), built
 *           as a dedicated block so that its area can be compared with the divider-based ibert_swiglu and with the share of the VPU
 *           that SwiGLU uses. Host-sequenced in 8-lane groups, like the controller:
 *             cmd 0  SILU   gate group x_in -> silu register: sigma_rom lookup, silu = sat12((x * sigma + 2) >> 2)   (4 cycles to done)
 *             cmd 1  MAX    up group x_in: running max of |silu * up|
 *             cmd 2  GAIN   row gain e from the max (clears the max)
 *             cmd 3  APPLY  up group x_in: y = sat12(round_shift((silu * up) << 2, 10 - e))
 *           Pass 1 of a row is SILU + MAX per group, then GAIN; pass 2 is SILU + APPLY per group. out_valid pulses when a command has
 *           completed; scale_o is the row gain e (signed). MAX / GAIN / APPLY are swiglu_core.
 */
module tr_swiglu #(
    parameter int N     = 8,
    parameter int W     = 8,
    parameter int OUT_W = 12
)(
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  in_valid,
    input  logic [1:0]            cmd,
    input  logic signed [W-1:0]   x_in [N],
    output logic                  out_valid,
    output logic signed [OUT_W-1:0] y_out [N],
    output logic [15:0]           scale_o
);

    localparam int SW_W = 12;

    // ---------------- silu of the gate group ----------------
    logic                  s1_v, s2_v, s3_v;
    logic signed [W-1:0]   g1 [N], g2 [N];
    logic [6:0]            sig_c [N], sig_r [N];
    logic signed [SW_W-1:0] silu_c [N];

    sigma_rom #(.N(N)) u_sigma (.x(g1), .sigma(sig_c));

    always_comb
        for (int i = 0; i < N; i++) begin
            logic signed [W+7:0] w;
            logic signed [W+7:0] s;
            w = g2[i] * $signed({1'b0, sig_r[i]});
            s = (w + (W+8)'(2)) >>> 2;
            silu_c[i] = (s > 2047) ? SW_W'(2047) : (s < -2048) ? SW_W'(-2048) : SW_W'(s);
        end

    // ---------------- back end ----------------
    logic                   core_done, silu_done;
    logic signed [4:0]      e_o;
    swiglu_core #(.N(N), .W(W), .SW_W(SW_W), .OUT_W(OUT_W)) u_core (
        .clk(clk), .rst_n(rst_n),
        .silu_load(s2_v), .silu_in(silu_c),
        .op_valid(in_valid && cmd != 2'd0), .op(2'(cmd - 2'd1)), .up_in(x_in),
        .op_done(core_done), .y_out(y_out), .e_o(e_o)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_v <= 1'b0; s2_v <= 1'b0; silu_done <= 1'b0; out_valid <= 1'b0;
            for (int i = 0; i < N; i++) begin g1[i] <= '0; g2[i] <= '0; sig_r[i] <= '0; end
        end else begin
            s1_v <= in_valid && (cmd == 2'd0);
            if (in_valid && cmd == 2'd0) g1 <= x_in;
            s2_v <= s1_v;
            if (s1_v) begin g2 <= g1; sig_r <= sig_c; end
            silu_done <= s2_v;
            out_valid <= silu_done | core_done;
        end
    end

    assign scale_o = 16'(e_o);

endmodule
