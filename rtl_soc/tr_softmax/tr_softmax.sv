`timescale 1ns/1ps

/*
 * @module   tr_softmax
 * @brief    INT8 quantized softmax using Taylor-region approximation
 * @details  Pass-based architecture driven externally by a controller.
 *           No inter-pass delay registers inside this module; pipeline
 *           latency lives only within the instantiated sub-modules.
 *
 *   mode 3'b000 — Pass 1 : MAX extraction        (piped_max)
 *   mode 3'b001 — Pass 2 : exp(xi−max) DOT sum   (scalar_sub → tr_exp_alu → vec_mul_dot)
 *   mode 3'b010 — Pass 3 : ln(S)                 (tr_ln_alu, combinational)
 *   mode 3'b011 — Pass 4 : exp(xi−max) ELEMWISE  (scalar_sub → tr_exp_alu → vec_mul_ew)
 *   mode 3'b100 — Pass 5 : exp(−ln(S)) = 1/S     (scalar_sub → tr_exp_alu → vec_mul_ew)
 *                          Controller sets x_in[i]=0, offset_in=ln(S) so x_sub = −ln(S)
 *   mode 3'b101 — Pass 6 : exp(xi−max) × 1/S     (vec_mul_ew direct, SS multiply)
 *                          Controller sets x_in=stored exp values, offset_in=1/S
 *
 * @param    N               Number of parallel lanes
 * @param    W               Data width (default 8-bit)
 * @param    FRAC_W          Fractional bits in Q format (default 4 → Q4.4)
 * @param    ACC_W           Accumulator width for vec_mul
 */
module tr_softmax #(
    parameter int N      = 8,
    parameter int W      = 8,
    parameter int FRAC_W = 4,
    parameter int ACC_W  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic [2:0]               mode,
    input  logic signed [W-1:0]      x_in      [N],
    input  logic signed [W-1:0]      offset_in,     // max(X) for passes 2/4/5; 1/S for pass 6
    input  logic signed [ACC_W-1:0]  sum_in,         // S (accumulator) for pass 3
    output logic                     valid_out,
    output logic signed [W-1:0]      y_out     [N],
    output logic signed [ACC_W-1:0]  sum_out
);

    // ====================================================================
    // PASS 1: MAX EXTRACTION  (mode == 3'b000)
    // ====================================================================
    logic                max_valid;
    logic signed [W-1:0] max_out;

    piped_max #(.NUM_INPUTS(N), .DATA_WIDTH(W)) u_max (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (valid_in && (mode == 3'b000)),
        .in_data  (x_in),
        .max_out  (max_out),
        .valid_out(max_valid)
    );

    // ====================================================================
    // PASS 3: LOGARITHM  (mode == 3'b010)
    // sum_in[23:8] extracts the Q12.4 slice that tr_ln_alu expects.
    // Fully combinational — valid_out mirrors valid_in for this pass.
    // ====================================================================
    logic signed [W-1:0] ln_out;

    tr_ln_alu #(.WIDTH(16), .BITS(4), .OUT_WIDTH(W)) u_ln (
        .xq(sum_in[23:8]),
        .yq(ln_out)
    );

    // ====================================================================
    // EXPONENTIAL DATAPATH  (modes 001, 011, 100)
    // x_sub = x_in − offset_in  (combinational saturation)
    // N parallel tr_exp_alu units share one LUT ROM
    // ====================================================================
    logic signed [W-1:0] x_sub [N];

    scalar_sub #(.NUM_INPUTS(N), .DATA_WIDTH(W)) u_sub (
        .in_data (x_in),
        .sub_val (offset_in),
        .out_data(x_sub)
    );

    logic [2:0]   a_idx   [N];
    logic [W-1:0] mantisa [N];
    logic [N-1:0] is_zero;

    genvar gi;
    generate
        for (gi = 0; gi < N; gi++) begin : gen_exp
            tr_exp_alu #(
                .WIDTH(W), .FRAC_W(FRAC_W), .LUT_IDX_W(3), .ITER(2)
            ) u_exp (
                .x       (x_sub[gi]),
                .a_idx   (a_idx[gi]),
                .mantisa (mantisa[gi]),
                .is_zero (is_zero[gi])
            );
        end
    endgenerate

    logic [N-1:0] e_a [N];

    shared_lut_rom #(.N(N)) u_lut (.a_idx(a_idx), .e_a(e_a));

    // ====================================================================
    // vec_mul INPUT MUX
    // is_zero (rounded_mag==0, fires for x_sub ∈ {−8,...,0}): use E=255
    //   matching C++ EXP_LUT_CONST[8]; ALU mantisa is used as-is.
    //   For x_sub==0 specifically: mantisa=16, so product=255×16=4080,
    //   [15:8]=15, which correctly mirrors C++ tr_approx_exp_scalar(0)=255.
    // Pass 6 feeds raw exp values and scalar 1/S directly, skipping exp ALU.
    // ====================================================================
    logic [W-1:0] dot_a [N], dot_b [N];
    logic [W-1:0] ew_a  [N], ew_b  [N];

    always_comb begin
        for (int j = 0; j < N; j++) begin
            dot_a[j] = is_zero[j] ? W'(255) : e_a[j];
            dot_b[j] = mantisa[j];

            if (mode == 3'b101) begin
                // Pass 6: direct multiply of stored exp × broadcast 1/S
                ew_a[j] = x_in[j];
                ew_b[j] = offset_in;
            end else begin
                ew_a[j] = is_zero[j] ? W'(255) : e_a[j];
                ew_b[j] = mantisa[j];
            end
        end
    end

    // ====================================================================
    // DOT PRODUCT vec_mul  — Pass 2: S = Σ exp(xi − max)
    // Fixed DOT mode; always clears the accumulator (one-shot per pass).
    // ====================================================================
    logic                    dot_valid;
    logic [N-1:0]            dot_mask;
    logic signed [ACC_W-1:0] dot_out [N];

    vec_mul #(.N(N), .W(W), .ACC_W(ACC_W)) u_vec_mul_dot (
        .clk           (clk),
        .rst_n         (rst_n),
        .in_valid      (valid_in && (mode == 3'b001)),
        .in_ready      (),
        .op_mode       (2'd2),    // UU: unsigned × unsigned
        .mode_elemwise (1'b0),    // DOT product, fixed
        .a             (dot_a),
        .b             (dot_b),
        .clear_acc     (1'b1),    // Fresh accumulation each pass
        .out_valid     (dot_valid),
        .out_ready     (1'b1),
        .out_valid_mask(dot_mask),
        .out_vec       (dot_out)
    );

    // ====================================================================
    // ELEMWISE vec_mul  — Passes 4 (exp values), 5 (1/S), 6 (probabilities)
    // Fixed ELEMWISE mode; op_mode selected per-pass at input capture time.
    // ====================================================================
    logic                    ew_valid;
    logic signed [ACC_W-1:0] ew_out [N];

    vec_mul #(.N(N), .W(W), .ACC_W(ACC_W)) u_vec_mul_ew (
        .clk           (clk),
        .rst_n         (rst_n),
        .in_valid      (valid_in && (mode == 3'b011 || mode == 3'b100 || mode == 3'b101)),
        .in_ready      (),
        // UU for all passes: pass 6 now multiplies two Q0.8 UNSIGNED magnitudes
        // (see output mux below), so SS would corrupt any operand >= 128 by
        // reading its top bit as a sign.
        .op_mode       (2'd2),
        .mode_elemwise (1'b1),                              // ELEMWISE, fixed
        .a             (ew_a),
        .b             (ew_b),
        .clear_acc     (1'b1),
        .out_valid     (ew_valid),
        .out_ready     (1'b1),
        .out_valid_mask(),
        .out_vec       (ew_out)
    );

    // ====================================================================
    // OUTPUT MUX
    // The controller holds mode stable while waiting for valid_out,
    // so mode here correctly identifies which pass produced the result.
    // ====================================================================
    always_comb begin
        valid_out = 1'b0;
        sum_out   = '0;
        y_out     = '{default: '0};

        case (mode)
            3'b000: begin
                // Pass 1: scalar max
                valid_out = max_valid;
                y_out[0]  = max_out;
            end
            3'b001: begin
                // Pass 2: dot-product sum S
                valid_out = dot_valid && dot_mask[0];
                sum_out   = dot_out[0];
            end
            3'b010: begin
                // Pass 3: ln(S) — combinational output, valid immediately
                valid_out = valid_in;
                y_out[0]  = ln_out;
            end
            3'b011, 3'b100: begin
                // Pass 4/5: E(Q0.8) x mantisa(Q4.4) product is scaled by 4096;
                // >>>4 lands on Q0.8 UNSIGNED (scale 256) instead of the old
                // >>>8 Q4.4 (scale 16) -- these values (exp(xi-max) and 1/S)
                // are always in [0,~1], so Q0.8 keeps 16x the useful
                // resolution. Saturate at 255 rather than wrap (the is_zero
                // anchor, E=255, can push the shifted product past 255).
                valid_out = ew_valid;
                for (int j = 0; j < N; j++) begin
                    logic signed [ACC_W-1:0] shifted;
                    shifted  = ew_out[j] >>> 4;
                    y_out[j] = (shifted > 255) ? W'(255) : W'(shifted);
                end
            end
            3'b101: begin
                // Pass 6: Q0.8 UNSIGNED x Q0.8 UNSIGNED -> Q0.16 product (scale
                // 65536); >>>8 extracts Q0.8 UNSIGNED (scale 256) directly --
                // the whole probability already lives in the low byte since it
                // never exceeds 1.0. Saturate at 255 defensively (max observed
                // product is 15*15=225, well under the 255 ceiling, but this
                // keeps the same safety margin as passes 4/5 above).
                valid_out = ew_valid;
                for (int j = 0; j < N; j++) begin
                    logic signed [ACC_W-1:0] shifted;
                    shifted  = ew_out[j] >>> 8;
                    y_out[j] = (shifted > 255) ? W'(255) : W'(shifted);
                end
            end
            default: valid_out = 1'b0;
        endcase
    end

endmodule
