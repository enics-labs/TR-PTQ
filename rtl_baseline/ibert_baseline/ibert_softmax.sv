`timescale 1ns/1ps

/*
 * @module   ibert_softmax
 * @brief    Standalone divider-based softmax tile: I-BERT integer exponential (Algorithm 3) and one integer division per lane.
 * @details  Golden: tr_math_model.hpp ibert_softmax_model. One command per 8-lane tile (in_valid with x_in):
 *             max = max over the lanes;  d_i = sat8(x_i - max)
 *             a = 2 |d_i| (one extra input fractional bit, scale 1/32);  z = a / 22;  p = 22 z - a;  e_i = ((p + 43)^2 + 983) >> z
 *             S = sum e_i;   p_i = min(255, (256 e_i + S / 2) / S)    -- eight sequential divisions through one restoring divider
 *           Same exponential constants as ibert_swiglu. The output is Q0.8 unsigned like tr_softmax; there is no log-partition value (a tile
 *           is normalized by its own sum). The divider is ibert_divider.sv (WIDTH cycles per division). out_valid pulses when p_out is valid.
 */
module ibert_softmax #(
    parameter int N = 8,
    parameter int W = 8
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 in_valid,
    input  logic signed [W-1:0]  x_in [N],
    output logic                 out_valid,
    output logic [W-1:0]         p_out [N]
);

    localparam int XLN2  = 22;
    localparam int QB    = 43;
    localparam int QC    = 983;
    localparam int E_W   = 12;                           // e <= 2832
    localparam int S_W   = E_W + $clog2(N);              // sum <= 22656
    localparam int DIV_W = 20;                           // 256 e + S / 2 < 2^20

    typedef enum logic [3:0] {IDLE, MAXF, SUB, EXPO, SUM, ISSUE, WAIT, STEP, DONE} state_t;
    state_t state;

    logic signed [W-1:0] x_r [N];
    logic signed [W-1:0] max_r;
    logic signed [W-1:0] d_r [N];
    logic [E_W-1:0]      e_r [N];
    logic [S_W-1:0]      S_r;
    logic [W-1:0]        p_r [N];
    logic [3:0]          lane;

    logic signed [W-1:0] max_c;
    always_comb begin
        logic signed [W-1:0] t [N];
        for (int i = 0; i < N; i++) t[i] = x_r[i];
        for (int lv = 0; lv < $clog2(N); lv++)
            for (int i = 0; i < (N >> (lv + 1)); i++) t[i] = (t[2*i] > t[2*i+1]) ? t[2*i] : t[2*i+1];
        max_c = t[0];
    end

    logic signed [W-1:0] d_c [N];
    always_comb
        for (int i = 0; i < N; i++) begin
            logic signed [W:0] diff;
            diff   = (W+1)'(x_r[i]) - (W+1)'(max_r);
            d_c[i] = (diff < -128) ? W'(-128) : W'(diff[W-1:0]);     // diff <= 0
        end

    // exponential of d (<= 0) per lane
    logic [E_W-1:0] e_c [N];
    always_comb
        for (int i = 0; i < N; i++) begin
            logic signed [W:0]  ds;
            logic [W-1:0]       ad;
            logic [W:0]         a;
            logic [3:0]         z;
            logic signed [15:0] p, t, poly;
            ds   = (W+1)'(d_r[i]);
            ad   = W'(ds[W] ? -ds : ds);
            a    = {ad, 1'b0};
            z    = 4'(a / XLN2);
            p    = 16'(XLN2 * z) - 16'({1'b0, a});
            t    = p + 16'(QB);
            poly = t * t + 16'(QC);
            e_c[i] = E_W'(poly >>> z);
        end

    logic [S_W-1:0] sum_c;
    always_comb begin
        logic [S_W-1:0] t [N];
        for (int i = 0; i < N; i++) t[i] = S_W'(e_r[i]);
        for (int lv = 0; lv < $clog2(N); lv++)
            for (int i = 0; i < (N >> (lv + 1)); i++) t[i] = t[2*i] + t[2*i+1];
        sum_c = t[0];
    end

    logic               div_valid_in, div_valid_out;
    logic [DIV_W-1:0]   div_dividend, div_divisor, div_quotient;
    assign div_dividend = DIV_W'({e_r[lane], 8'b0} + (S_r >> 1));
    assign div_divisor  = DIV_W'(S_r);

    ibert_divider #(.WIDTH(DIV_W)) u_div (
        .clk(clk), .rst_n(rst_n), .valid_in(div_valid_in), .dividend(div_dividend), .divisor(div_divisor),
        .busy(), .valid_out(div_valid_out), .quotient(div_quotient), .remainder()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; out_valid <= 1'b0; div_valid_in <= 1'b0; max_r <= '0; S_r <= '0; lane <= '0;
            for (int i = 0; i < N; i++) begin x_r[i] <= '0; d_r[i] <= '0; e_r[i] <= '0; p_r[i] <= '0; p_out[i] <= '0; end
        end else begin
            out_valid    <= 1'b0;
            div_valid_in <= 1'b0;
            case (state)
                IDLE:  if (in_valid) begin x_r <= x_in; state <= MAXF; end
                MAXF:  begin max_r <= max_c; state <= SUB; end
                SUB:   begin d_r <= d_c; state <= EXPO; end
                EXPO:  begin e_r <= e_c; state <= SUM; end
                SUM:   begin S_r <= sum_c; lane <= '0; state <= ISSUE; end
                ISSUE: begin div_valid_in <= 1'b1; state <= WAIT; end
                WAIT:  if (div_valid_out) state <= STEP;
                STEP: begin
                    p_r[lane] <= (div_quotient > 255) ? 8'd255 : W'(div_quotient);
                    if (lane == N - 1) state <= DONE;
                    else begin lane <= lane + 1'b1; state <= ISSUE; end
                end
                DONE:  begin p_out <= p_r; out_valid <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
