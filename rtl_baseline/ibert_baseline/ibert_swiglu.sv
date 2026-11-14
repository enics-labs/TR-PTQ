`timescale 1ns/1ps

/*
 * @module   ibert_swiglu
 * @brief    Standalone divider-based SwiGLU: the sigmoid of the gate code from the I-BERT integer exponential and one integer division
 *           per lane, with the same 12-bit silu and output codes as tr_swiglu.
 * @details  Same command interface and back end (swiglu_core) as tr_swiglu, so the two differ only in how the silu of a gate group is
 *           produced. Golden: tr_math_model.hpp ibert_sigma6, ibert_silu (then sw_abs_prod, sw2_gain, sw2_apply as in tr_swiglu).
 *             cmd 0  SILU   per lane, sequentially through one restoring divider (ibert_divider.sv, WIDTH cycles per division):
 *                           a = 2 |x|; z = a / 22; p = z * 22 - a; e = ((p + 43)^2 + 983) >> z          (I-BERT Algorithm 3, scale 1/32)
 *                           sigma = (N * 64 + D / 2) / D with D = 2859 + e, N = 2859 (x >= 0) or e (x < 0);  silu = sat12((x * sigma + 2) >> 2)
 *             cmd 1..3  MAX / GAIN / APPLY   as in tr_swiglu
 *           The divisions by the constant 22 and the polynomial are combinational per lane; only the sigma division is iterative.
 *           out_valid pulses when a command has completed; scale_o is the row gain e (signed).
 */
module ibert_swiglu #(
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

    localparam int SW_W   = 12;
    localparam int XLN2   = 22;
    localparam int QB     = 43;
    localparam int QC     = 983;
    localparam int ONE    = 2859;
    localparam int FRAC   = 6;
    localparam int DIV_W  = 18;                          // N * 64 + D / 2 < 2^18

    typedef enum logic [2:0] {IDLE, PREP, ISSUE, WAIT, STEP, LOAD, CORE_WAIT} state_t;
    state_t state;

    logic signed [W-1:0]    xr [N];
    logic signed [SW_W-1:0] silu_v [N];
    logic [3:0]             lane;

    // ---------------- exponential of -|x| for the lane being processed ----------------
    logic signed [W:0]  xs_c;                            // gate code, sign-extended
    logic [W-1:0]       ax_c;                            // |x| (<= 128)
    logic [W:0]         a_c;                             // 2 |x| (<= 256)
    logic [3:0]         z_c;
    logic signed [15:0] p_c, t_c, poly_c;
    logic [11:0]        e_c, n_c;                        // e <= 2832, N <= 2859
    logic [12:0]        d_c;                             // D = 2859 + e <= 5691
    always_comb begin
        xs_c   = (W+1)'(xr[lane]);
        ax_c   = W'(xs_c[W] ? -xs_c : xs_c);
        a_c    = {ax_c, 1'b0};
        z_c    = 4'(a_c / XLN2);
        p_c    = 16'(XLN2 * z_c) - 16'({1'b0, a_c});     // in [-21, 0]
        t_c    = p_c + 16'(QB);
        poly_c = t_c * t_c + 16'(QC);
        e_c    = 12'(poly_c >>> z_c);
        d_c    = 13'(ONE) + 13'(e_c);
        n_c    = xr[lane][W-1] ? e_c : 12'(ONE);
    end

    logic [11:0]        n_r;
    logic [12:0]        d_r;
    logic               div_valid_in, div_valid_out;
    logic [DIV_W-1:0]   div_dividend, div_divisor, div_quotient;
    assign div_dividend = DIV_W'({n_r, 6'b0} + (d_r >> 1));
    assign div_divisor  = DIV_W'(d_r);

    ibert_divider #(.WIDTH(DIV_W)) u_div (
        .clk(clk), .rst_n(rst_n), .valid_in(div_valid_in), .dividend(div_dividend), .divisor(div_divisor),
        .busy(), .valid_out(div_valid_out), .quotient(div_quotient), .remainder()
    );

    // silu of the lane from the quotient (sigma, Q.6)
    logic signed [SW_W-1:0] silu_c;
    always_comb begin
        logic signed [W+7:0] w, s;
        w = xr[lane] * $signed({1'b0, div_quotient[6:0]});
        s = (w + (W+8)'(2)) >>> 2;
        silu_c = (s > 2047) ? SW_W'(2047) : (s < -2048) ? SW_W'(-2048) : SW_W'(s);
    end

    // ---------------- back end ----------------
    logic                   core_done, load_pulse;
    logic signed [4:0]      e_o;
    assign load_pulse = (state == LOAD);
    swiglu_core #(.N(N), .W(W), .SW_W(SW_W), .OUT_W(OUT_W)) u_core (
        .clk(clk), .rst_n(rst_n),
        .silu_load(load_pulse), .silu_in(silu_v),
        .op_valid(in_valid && cmd != 2'd0 && state == IDLE), .op(2'(cmd - 2'd1)), .up_in(x_in),
        .op_done(core_done), .y_out(y_out), .e_o(e_o)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; out_valid <= 1'b0; div_valid_in <= 1'b0; lane <= '0; n_r <= '0; d_r <= '0;
            for (int i = 0; i < N; i++) begin xr[i] <= '0; silu_v[i] <= '0; end
        end else begin
            out_valid    <= 1'b0;
            div_valid_in <= 1'b0;
            case (state)
                IDLE: if (in_valid) begin
                    if (cmd == 2'd0) begin xr <= x_in; lane <= '0; state <= PREP; end
                    else             state <= CORE_WAIT;
                end
                PREP:  begin n_r <= n_c; d_r <= d_c; state <= ISSUE; end
                ISSUE: begin div_valid_in <= 1'b1; state <= WAIT; end
                WAIT:  if (div_valid_out) state <= STEP;
                STEP: begin
                    silu_v[lane] <= silu_c;
                    if (lane == N - 1) state <= LOAD;
                    else begin lane <= lane + 1'b1; state <= PREP; end
                end
                LOAD: begin out_valid <= 1'b1; state <= IDLE; end
                CORE_WAIT: if (core_done) begin out_valid <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

    assign scale_o = 16'(e_o);

endmodule
