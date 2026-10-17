`ifndef TR_GELU_EN_DEFAULT
`ifdef GELU_ENABLE
`define TR_GELU_EN_DEFAULT 1'b1   // opt-in: GELU support (ViT-style builds). Default: no GELU (GELU and SwiGLU never run together)
`else
`define TR_GELU_EN_DEFAULT 1'b0
`endif
`endif

/*
 * @module   vecmul_out_cast
 * @brief    Output stage of tr_nonlinear_vpu's vec-mul engine: scales / rounds / saturates the N wide products to W bits.
 * @details  Combinational. vecmul_scale_mode selects the cast (these are the encodings the controller drives):
 *             000  low W bits, no scaling
 *             001  >>> FRAC_W, wrapping                       (GELU only)
 *             010  >>> 8                                      (GELU only; anchor Q0.8)
 *             011  softmax probabilities: >>> FRAC_W, unsigned saturate at 255
 *             101  RMSNorm apply: ROUNDED >>> (FRAC_W+4), signed saturate
 *             110  SwiGLU apply: round_shift(acc << 2, sw_shift) with sw_shift >= 0, signed saturate
 *           Split out of tr_nonlinear_vpu so synthesis reports its area on its own.
 *
 * @param    N        Lanes.
 * @param    W        Output width.
 * @param    ACC_W    Width of the engine's result (acc); the VPU passes 20 (the largest result is 20 signed bits).
 * @param    FRAC_W   Fractional bits of the Q-format.
 * @param    GELU_EN  0 (default; macro GELU_ENABLE sets 1): modes 001 / 010 are not built.
 * @param    EN_SOFTMAX / EN_RMS / EN_SWIGLU  1 (default): modes 011 / 101 / 110 are built. 0 removes that mode (it falls
 *                    back to the low-bits pass-through): for per-mode area breakdowns only, the controller needs all three.
 */
module vecmul_out_cast #(
    parameter int N       = 8,
    parameter int W       = 8,
    parameter int ACC_W   = 32,
    parameter int FRAC_W  = 4,
    parameter bit GELU_EN = `TR_GELU_EN_DEFAULT,
    parameter bit EN_SOFTMAX = 1'b1,
    parameter bit EN_RMS     = 1'b1,
    parameter bit EN_SWIGLU  = 1'b1
)(
    input  logic signed [ACC_W-1:0] acc [N],
    input  logic [2:0]              scale_mode,
    input  logic signed [5:0]       sw_shift,
    output logic signed [W-1:0]     out [N]
);

    localparam signed [W-1:0] SAT_MAX = (1 <<< (W-1)) - 1;
    localparam signed [W-1:0] SAT_MIN = -(1 <<< (W-1));

    always_comb begin
        for(int i=0; i<N; i++) begin
            logic signed [ACC_W-1:0] shifted;
            if (EN_SOFTMAX && scale_mode == 3'b011) begin
                // Softmax Q0.8 UNSIGNED probability output: shift = anchor_q_bits(8) + FRAC_W - 8 = FRAC_W.
                // Saturate at 255 (~0.996) instead of wrapping (a single dominant logit can land exactly on 256).
                shifted = acc[i] >>> FRAC_W;
                out[i] = (shifted > 255) ? W'(255) : W'(shifted);
            end else if (EN_SWIGLU && scale_mode == 3'b110) begin
                // SwiGLU apply: prod = acc << 2 (silu was stored with 2 dropped bits), requantized with the per-row
                // gain: ROUNDED right shift by sw_shift (always >= 0: the gain is capped at 10), signed saturate.
                // round_shift(p, s) = ((2p >>> s) + 1) >>> 1 (round half up, identical to (p + 2^(s-1)) >>> s and to p for
                // s = 0), so the shifter comes first and the +1 acts on the shifted value: no (s-1) subtract and no
                // wide adder in front of the shifter (timing).
                logic signed [ACC_W+2:0] p2;       // 2 * (acc << 2) = acc << 3
                logic signed [ACC_W+2:0] z;
                logic signed [ACC_W+2:0] r;
                p2 = $signed({{3{acc[i][ACC_W-1]}}, acc[i]}) <<< 3;
                z  = p2 >>> sw_shift[4:0];
                r  = (z + 1) >>> 1;
                if (r > SAT_MAX)        out[i] = SAT_MAX;
                else if (r < SAT_MIN)   out[i] = SAT_MIN;
                else                    out[i] = W'(r);
            end else if (EN_RMS && scale_mode == 3'b101) begin
                // RMSNorm apply: x * inv_rms (inv_rms has FRAC_W+4 fractional bits) -> Q.FRAC_W, ROUNDED, signed saturate.
                shifted = (acc[i] + (1 <<< (FRAC_W + 3))) >>> (FRAC_W + 4);
                if (shifted > SAT_MAX)       out[i] = SAT_MAX;
                else if (shifted < SAT_MIN)  out[i] = SAT_MIN;
                else                         out[i] = W'(shifted);
            end else if (GELU_EN && scale_mode == 3'b010) begin
                out[i] = W'(acc[i] >>> 8);
            end else if (GELU_EN && scale_mode == 3'b001) begin
                out[i] = W'(acc[i] >>> FRAC_W);
            end else begin
                out[i] = W'(acc[i]);
            end
        end
    end

endmodule


/*
 * @module   vecmul_out_cast_reg
 * @brief    PROFILING WRAPPER (not instantiated anywhere in the design): vecmul_out_cast between input and output registers.
 * @details  A combinational module synthesized on its own sees only the in2out constraints (input/output delay of 30% of the
 *           clock each), which it cannot meet, so the tool upsizes it and the area is not representative. In the VPU the cast sits
 *           between the vec-mul engine's output register and the vpu_data_out register; this wrapper reproduces that, so a standalone
 *           synthesis (TOPLEVEL=vecmul_out_cast_reg) gets a normal reg2reg path. The cast itself is reported as instance u_cast; the
 *           wrapper's own flops are a constant offset. Per-mode breakdown: elaborate with -parameters {{EN_SOFTMAX 0}}, {{EN_RMS 0}},
 *           {{EN_SWIGLU 0}} (and all three 0 for the fixed overhead). Defaults match how the VPU instantiates the cast (ACC_W = 20).
 */
module vecmul_out_cast_reg #(
    parameter int N       = 8,
    parameter int W       = 8,
    parameter int ACC_W   = 20,
    parameter int FRAC_W  = 4,
    parameter bit GELU_EN = `TR_GELU_EN_DEFAULT,
    parameter bit EN_SOFTMAX = 1'b1,
    parameter bit EN_RMS     = 1'b1,
    parameter bit EN_SWIGLU  = 1'b1
)(
    input  logic clk,
    input  logic rst_n,
    input  logic signed [ACC_W-1:0] acc [N],
    input  logic [2:0]              scale_mode,
    input  logic signed [5:0]       sw_shift,
    output logic signed [W-1:0]     out [N]
);

    logic signed [ACC_W-1:0] acc_q [N];
    logic [2:0]              mode_q;
    logic signed [5:0]       shift_q;
    logic signed [W-1:0]     out_d [N];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mode_q  <= '0;
            shift_q <= '0;
            for (int i = 0; i < N; i++) begin acc_q[i] <= '0; out[i] <= '0; end
        end else begin
            mode_q  <= scale_mode;
            shift_q <= sw_shift;
            for (int i = 0; i < N; i++) begin acc_q[i] <= acc[i]; out[i] <= out_d[i]; end
        end
    end

    vecmul_out_cast #(
        .N(N), .W(W), .ACC_W(ACC_W), .FRAC_W(FRAC_W), .GELU_EN(GELU_EN),
        .EN_SOFTMAX(EN_SOFTMAX), .EN_RMS(EN_RMS), .EN_SWIGLU(EN_SWIGLU)
    ) u_cast (
        .acc(acc_q), .scale_mode(mode_q), .sw_shift(shift_q), .out(out_d)
    );

endmodule
