`timescale 1ns/1ps

/*
 * @module   tr_rmsnorm
 * @brief    Standalone RMSNorm with the Taylor-Region reciprocal-sqrt unit (12-bit Q5.7 output), without the shared VPU backbone.
 * @details  The same arithmetic as the production flow of tr_soc_ctrl_int / tr_nonlinear_vpu (golden: tr_math_model.hpp rms_acc_tile,
 *           rms2_scale, rms2_apply_tile), built as a dedicated block so that its area can be compared with the standalone divider
 *           divider-based ibert_rmsnorm and with the share of the VPU that RMSNorm uses. Three host-sequenced commands, like the controller:
 *             cmd 0  ACC_FIRST   S  = sum x^2 of the tile          (4 cycles to done)
 *             cmd 1  ACC         S += sum x^2 of the tile
 *             cmd 2  SCALE       mk = round(S >> (log2n + 2)) saturated to 16 bits; leading-one finder and normalizer (msb, fk);
 *                                rsq_unit table + shift -> inv register   (4 cycles to done)
 *             cmd 3  APPLY       y = sat12((x * inv + 16) >> 5) per lane   (3 cycles to done)
 *           out_valid pulses when the command has completed. scale_o is the inv register. log2n must be stable from ACC to SCALE.
 *           The leading-one finder and normalizer here are the ones tr_ln_alu provides to rsq_unit in the VPU (msb = k, fk = the BITS
 *           mantissa bits below the leading one, truncated).
 */
module tr_rmsnorm #(
    parameter int N     = 8,
    parameter int W     = 8,
    parameter int OUT_W = 12,
    parameter int INV_W = 12,
    parameter int ACC_W = 32,
    parameter int MK_W  = 16,
    parameter int BITS  = 6
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

    localparam logic [1:0] CMD_ACC_FIRST = 2'd0, CMD_ACC = 2'd1, CMD_SCALE = 2'd2, CMD_APPLY = 2'd3;
    localparam int MSB_W = $clog2(MK_W);

    // ---------------- accumulate ----------------
    logic [ACC_W-1:0] S;
    logic             ss_done;
    rmsnorm_sumsq #(.N(N), .W(W), .ACC_W(ACC_W)) u_ss (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid && !cmd[1]), .in_first(cmd == CMD_ACC_FIRST),
        .x_in(x_in), .done(ss_done), .S(S)
    );

    // ---------------- scale ----------------
    logic                sc1_v, sc2_v, sc3_v, sc_done;
    logic [MK_W-1:0]     mk_r;
    logic [MSB_W-1:0]    msb_r;
    logic [BITS-1:0]     fk_r;
    logic [INV_W-1:0]    inv_r, inv_c;

    logic [4:0]          sh;
    logic [ACC_W:0]      mk_wide;
    logic [MK_W-1:0]     mk_c;
    always_comb begin
        sh      = 5'(log2n) + 5'd2;                                   // FRAC_W + log2n - K, FRAC_W = 4, K = 2
        mk_wide = ({1'b0, S} + ((ACC_W+1)'(1) << (sh - 1))) >> sh;
        mk_c    = (mk_wide > (ACC_W+1)'((1 << MK_W) - 1)) ? MK_W'((1 << MK_W) - 1) : mk_wide[MK_W-1:0];
    end

    logic [MSB_W-1:0]    msb_c;
    logic [BITS-1:0]     fk_c;
    logic [MK_W+BITS-1:0] norm_c;
    always_comb begin
        msb_c = '0;
        for (int k = 0; k < MK_W; k++) if (mk_r[k]) msb_c = MSB_W'(k);
        norm_c = {mk_r, {BITS{1'b0}}} >> msb_c;
        fk_c   = norm_c[BITS-1:0];                                    // mk = 0 -> msb = fk = 0 (reads like mk = 1)
    end

    rsq_unit #(.MSB_W(MSB_W), .BITS(BITS), .INV_W(INV_W)) u_rsq (.msb(msb_r), .fk(fk_r), .inv(inv_c));

    // ---------------- apply ----------------
    localparam int P_W = W + INV_W + 1;
    logic                    ap1_v, ap2_v, ap_done;
    logic signed [W-1:0]     xa [N];
    logic signed [P_W-1:0]   prod [N];
    logic signed [OUT_W-1:0] y_c [N];
    localparam int YMAX = (1 << (OUT_W-1)) - 1;
    localparam int YMIN = -(1 << (OUT_W-1));
    always_comb
        for (int i = 0; i < N; i++) begin
            logic signed [P_W-1:0] t;
            t = (prod[i] + P_W'(16)) >>> 5;                           // round half up, FRAC_W + INV_BITS + 4 - 7 = 5
            y_c[i] = (t > YMAX) ? OUT_W'(YMAX) : (t < YMIN) ? OUT_W'(YMIN) : OUT_W'(t);
        end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sc1_v <= 1'b0; sc2_v <= 1'b0; sc3_v <= 1'b0; sc_done <= 1'b0;
            mk_r <= '0; msb_r <= '0; fk_r <= '0; inv_r <= '0;
            ap1_v <= 1'b0; ap2_v <= 1'b0; ap_done <= 1'b0;
            for (int i = 0; i < N; i++) begin xa[i] <= '0; prod[i] <= '0; y_out[i] <= '0; end
            out_valid <= 1'b0;
        end else begin
            sc1_v <= in_valid && (cmd == CMD_SCALE);
            sc2_v <= sc1_v;
            sc3_v <= sc2_v;
            sc_done <= sc3_v;
            if (sc1_v) mk_r <= mk_c;
            if (sc2_v) begin msb_r <= msb_c; fk_r <= fk_c; end
            if (sc3_v) inv_r <= inv_c;

            ap1_v <= in_valid && (cmd == CMD_APPLY);
            ap2_v <= ap1_v;
            ap_done <= ap2_v;
            if (in_valid && cmd == CMD_APPLY) xa <= x_in;
            if (ap1_v) for (int i = 0; i < N; i++) prod[i] <= xa[i] * $signed({1'b0, inv_r});
            if (ap2_v) y_out <= y_c;

            out_valid <= ss_done | sc_done | ap_done;
        end
    end

    assign scale_o = 16'(inv_r);

endmodule
