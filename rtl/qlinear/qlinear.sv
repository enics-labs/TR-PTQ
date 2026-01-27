module vec_mac_requant #(
    parameter int N     = 32,
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Input stream
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic signed [W-1:0]     a [N],
    input  logic signed [W-1:0]     b [N],
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
    // Internal signals
    // ------------------------------------------------------------
    logic                    mac_out_valid;
    logic signed [ACC_W-1:0] mac_out_dot;
    logic                    requant_in_ready;

    // Buffered requant parameters
    logic signed [31:0]      m_0_reg;
    logic signed [4:0]       f_shift_reg;
    logic signed [31:0]      bias_reg;

    // ------------------------------------------------------------
    // Handshake control
    // ------------------------------------------------------------
    // The MAC can only advance if the Requant unit is ready to accept data
    assign in_ready = requant_in_ready;

    // ------------------------------------------------------------
    // Vector MAC
    // ------------------------------------------------------------
    vec_mac_dsp48 #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) mac (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .in_ready  (),               // Handled by top-level in_ready
        .a         (a),
        .b         (b),
        .clear_acc (clear_acc),
        .out_valid (mac_out_valid),
        .out_ready (requant_in_ready), // Requant unit backpressures the MAC
        .out_dot   (mac_out_dot)
    );

    // ------------------------------------------------------------
    // Parameter buffer (latched once per vector)
    // ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_0_reg     <= '0;
            f_shift_reg <= '0;
            bias_reg    <= '0;
        end else if (in_valid && in_ready && clear_acc) begin
            m_0_reg     <= m_0;
            f_shift_reg <= f_shift;
            bias_reg    <= bias;
        end
    end

    // ------------------------------------------------------------
    // Requant Unit (Internal Pipeline Stage)
    // ------------------------------------------------------------
    // This module now contains the logic and the output registers
    requant_unit requant_inst (
        .clk       (clk),
        .rst_n     (rst_n),
        
        // Input from MAC
        .in_valid  (mac_out_valid),
        .in_ready  (requant_in_ready),
        .acc_sum   (mac_out_dot[31:0]),
        
        // Parameters
        .m_0       (m_0_reg),
        .f_shift   (f_shift_reg),
        .bias      (bias_reg),
        
        // Final Outputs
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_quant (out_q)
    );

endmodule


