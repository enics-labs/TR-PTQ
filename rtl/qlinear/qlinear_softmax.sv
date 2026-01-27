module vec_mac_requant #(
    parameter int N     = 32,
    parameter int W     = 8,
    parameter int ACC_W = 32,
    parameter int FRAC  = 4,   // kept for compatibility (not used by exp_x_minus_xmax as provided)
    parameter int ITER  = 2
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Control
    input  logic                    mode_sel,   // 0: Standard MAC (a*b), 1: Exp Mode (e_frac * e_a)

    // Input stream
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic signed [W-1:0]     a [N],      // Standard input A OR input X for Exp
    input  logic signed [W-1:0]     b [N],      // Standard input B (ignored in Exp mode)
    input  logic                    clear_acc,

    // Requant parameters
    input  logic signed [31:0]      m_0,
    input  logic signed [4:0]       f_shift,
    input  logic signed [31:0]      bias,

    // Output stream (INT8)
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic signed [7:0]       out_q
);

    // ------------------------------------------------------------
    // Exponent Logic (vector-level) & Mode Selection
    // ------------------------------------------------------------
    logic                          exp_valid_out;
    logic        [W-1:0]           exp_ea_u   [N];   // unsigned from exp block
    logic        [W-1:0]           exp_efrac_u[N];   // unsigned from exp block

    logic signed [W-1:0]           mac_a_in   [N];
    logic signed [W-1:0]           mac_b_in   [N];

    // Feed exp block with the vector X = a[]
    // Note: valid is qualified with in_ready so upstream must hold a[] stable when !in_ready.
    exp_x_minus_xmax #(
        .NUM_INPUTS  (N),
        .DATA_WIDTH  (W),
        .ITER        (ITER)
    ) u_exp_x_minus_xmax (
        .clk       (clk),
        .rst_n      (rst_n),
        .valid_in   (in_valid & in_ready),
        .in_data    (a),

        .valid_out  (exp_valid_out),
        .e_a        (exp_ea_u),
        .e_frac     (exp_efrac_u)
    );

    // MUX logic to select between original inputs or exponent outputs
    // In exp mode: MAC does e_frac * e_a (both are non-negative), but vec_mac is signed.
    // Cast to signed safely by forcing MSB=0 before $signed.
    generate
        for (genvar i = 0; i < N; i++) begin : gen_mux
            logic signed [W-1:0] exp_ea_s;
            logic signed [W-1:0] exp_efrac_s;

            // Force positive signed interpretation (assumes W is sufficient and values are intended unsigned)
            assign exp_ea_s    = $signed({1'b0, exp_ea_u[i][W-2:0]});
            assign exp_efrac_s = $signed({1'b0, exp_efrac_u[i][W-2:0]});

            assign mac_a_in[i] = mode_sel ? exp_efrac_s : a[i];
            assign mac_b_in[i] = mode_sel ? exp_ea_s    : b[i];
        end
    endgenerate

    // ------------------------------------------------------------
    // Pipeline Synchronization (clear_acc alignment for exp mode)
    // ------------------------------------------------------------
    // exp_x_minus_xmax contains a pipelined max-tree with latency:
    //   MAX_LATENCY = $clog2(N) - 1
    // and then an internal vector delay of MAX_LATENCY+1.
    // Your exp_x_minus_xmax currently sets valid_out using valid_d[MAX_LATENCY-1]
    // and notes it may need adjustment if tr_exp is pipelined.
    //
    // Here we conservatively delay clear_acc by EXP_LATENCY cycles, and you can tune if needed.
    localparam int EXP_LATENCY = (N <= 2) ? 1 : $clog2(N); // conservative default
    logic [EXP_LATENCY-1:0] clear_delay;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            clear_delay <= '0;
        end else if (in_ready) begin
            clear_delay <= {clear_delay[EXP_LATENCY-2:0], clear_acc};
        end
    end

    wire mac_in_valid = mode_sel ? exp_valid_out                 : in_valid;
    wire mac_clear    = mode_sel ? clear_delay[EXP_LATENCY-1]     : clear_acc;

    // ------------------------------------------------------------
    // Vector MAC
    // ------------------------------------------------------------
    logic                    mac_out_valid;
    logic signed [ACC_W-1:0] mac_out_dot;
    logic                    requant_in_ready;

    vec_mac_dsp48 #(.N(N), .W(W), .ACC_W(ACC_W)) mac (
        .clk(clk), .rst_n(rst_n),
        .in_valid(mac_in_valid),
        .in_ready(), 
        .a(mac_a_in), .b(mac_b_in),
        .clear_acc(mac_clear),
        .out_valid(mac_out_valid),
        .out_ready(requant_in_ready),
        .out_dot(mac_out_dot)
    );

    // ------------------------------------------------------------
    // Pipelined Requant Unit (3-Stage)
    // ------------------------------------------------------------
    logic signed [31:0] m_0_reg, bias_reg;
    logic signed [4:0]  f_shift_reg;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            {m_0_reg, f_shift_reg, bias_reg} <= '0;
        end else if (in_valid && in_ready && clear_acc) begin
            m_0_reg     <= m_0;
            f_shift_reg <= f_shift;
            bias_reg    <= bias;
        end
    end

    requant_unit requant_inst (
        .clk(clk), .rst_n(rst_n),
        .in_valid(mac_out_valid), .in_ready(requant_in_ready),
        .out_valid(out_valid), .out_ready(out_ready),
        .acc_sum(mac_out_dot[31:0]),
        .m_0(m_0_reg), .f_shift(f_shift_reg), .bias(bias_reg),
        .out_quant(out_q)
    );

    assign in_ready = requant_in_ready;

endmodule
