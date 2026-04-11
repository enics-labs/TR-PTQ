`timescale 1ns/1ps

module tr_hetero_array #(
    parameter int N = 8
)(
    input  logic               clk,
    input  logic               rst_n,

    // Data Inputs
    input  logic signed [7:0]  x_vec_in [N],   // Vector feed (from Memory or Max_Sub)
    input  logic signed [19:0] x_scalar_in,    // Scalar feed (from MAC out_dot)
    
    // Control Inputs
    input  logic               lane_0_mode, // 0: Vector (GELU), 1: Scalar (LN/SoftMax)
    input  logic [1:0]         shift_mode,  // 00: Bypass, 01: -x (1/x), 10: -(x>>>1) (1/sqrt(x))
    input  logic               exp_in_sel,  // 0: tr_exp takes x_vec_in, 1: tr_exp takes tr_ln output
    
    // Data Outputs
    output logic [7:0]         y_ea_vec_out [N],
    output logic [7:0]         y_man_vec_out [N],
    output logic [19:0]        y_ea_scalar_out,
    output logic [19:0]        y_man_scalar_out
);

    // ========================================================================
    // SHARED ROM INTERCONNECT (Used by all 8-bit ALUs)
    // ========================================================================
    logic [2:0] shared_idx    [N];
    logic [7:0] shared_ea_raw [N];
    
    shared_lut_rom #(.N(N)) u_shared_rom (
        .a_idx(shared_idx),
        .e_a(shared_ea_raw)
    );

    // ========================================================================
    // LANE 0: THE HETEROGENEOUS SUPER-LANE
    // ========================================================================
    // --------------------------------------------------------
    // PATH A: The High-Precision 20-Bit Processor
    // --------------------------------------------------------
    logic signed [19:0] l0_ln_out_20b, l0_shift_out_20b, l0_exp_in_20b, l0_man_20b;
    logic [3:0]         idx_20b;
    logic [19:0]        ea_20b_raw;
    logic               z_20b;

    logic signed [7:0] l0_ln_out_8b, l0_shift_out_8b, l0_exp_in_8b, l0_man_8b;
    logic              z_8b;

    // 1. The Logarithms
    tr_ln #(            // 20-bit Logarithm
        .WIDTH(20), 
        .BITS(8),
        .OUT_WIDTH(20)
    ) u_ln_0_20b (
        .xq(x_scalar_in), 
        .yq(l0_ln_out_20b)
    );
    
    tr_ln #(            // 8-bit Logarithm
        .WIDTH(8), 
        .BITS(4),
        .OUT_WIDTH(8)
    ) u_ln_0_8b (
        .xq(x_vec_in[0]), 
        .yq(l0_ln_out_8b)
    );
    
    // 2. The Intercept Shifter (Math Reflector)
    always_comb begin
        case (shift_mode)
            2'b01: begin
                l0_shift_out_20b = -l0_ln_out_20b;         // SoftMax: 1/x
                l0_shift_out_8b  = -l0_ln_out_8b;
            end
            2'b10:   begin
                l0_shift_out_20b = -(l0_ln_out_20b >>> 1); // LayerNorm: 1/sqrt(x)
                l0_shift_out_8b  = -(l0_ln_out_8b >>> 1);
            end
            default: begin
                l0_shift_out_20b = l0_ln_out_20b;          // Bypass
                l0_shift_out_8b  = l0_ln_out_8b;
            end
        endcase
    end

    // 3. Mux to choose between external data or the shifter output
    assign l0_exp_in_20b = exp_in_sel ? l0_shift_out_20b : x_scalar_in;
    assign l0_exp_in_8b  = exp_in_sel ? l0_shift_out_8b  : x_vec_in[0];

    // 4. Pipeline Registers
    logic signed [19:0] l0_exp_in_20b_pipe;
    logic signed [7:0]  l0_exp_in_8b_pipe;
    logic               lane_0_mode_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            l0_exp_in_20b_pipe <= '0;
            l0_exp_in_8b_pipe  <= '0;
            lane_0_mode_pipe   <= '0;
        end else begin
            l0_exp_in_20b_pipe <= l0_exp_in_20b;
            l0_exp_in_8b_pipe  <= l0_exp_in_8b;
            lane_0_mode_pipe   <= lane_0_mode;
        end
    end

    // 5.The Exponentials
    q12_8_lut u_lut_20b (   // 20-bit LUT
        .a_idx(idx_20b), 
        .e_a(ea_20b_raw)
    );
    
    tr_exp_alu #(           // 20-bit Exponential
        .WIDTH(20), 
        .FRAC_W(8), 
        .LUT_IDX_W(4)
    ) u_exp_0_20b (
        .x(l0_exp_in_20b_pipe), 
        .a_idx(idx_20b), 
        .mantisa(l0_man_20b),
        .is_zero(z_20b)
    );
    
    tr_exp_alu #(           // 8-bit Exponential
        .WIDTH(8), 
        .FRAC_W(4), 
        .LUT_IDX_W(3)
    ) u_exp_0_8b (
        .x(l0_exp_in_8b_pipe), 
        .a_idx(shared_idx[0]),  // Hooked to Shared ROM[0]
        .mantisa(l0_man_8b),
        .is_zero(z_8b)
    );

    // --------------------------------------------------------
    // LANE 0 OUTPUT MULTIPLEXING
    // --------------------------------------------------------
    // Apply the e^0 bypass directly to the anchors
    wire [19:0] actual_ea_20b = z_20b ? 20'd256 : ea_20b_raw;
    wire [7:0]  actual_ea_8b  = z_8b  ? 8'd255  : shared_ea_raw[0];

    assign y_ea_scalar_out  = actual_ea_20b;
    assign y_man_scalar_out = l0_man_20b; 
    
    assign y_ea_vec_out[0]  = lane_0_mode_pipe ? actual_ea_20b[11:4] : actual_ea_8b;
    assign y_man_vec_out[0] = lane_0_mode_pipe ? l0_man_20b[11:4]    : l0_man_8b;

    // ========================================================================
    // LANES 1 to N-1: STANDARD 8-BIT VECTOR PROCESSING
    // ========================================================================
    generate
        for (genvar i = 1; i < N; i++) begin : GEN_LANES
            logic signed [7:0] ln_out_i, shift_out_i, exp_in_i, exp_in_i_pipe;
            logic              z_i;
            
            // 1. Logarithm
            tr_ln #(
                .WIDTH(8), 
                .BITS(4),
                .OUT_WIDTH(8)
            ) u_ln_i (
                .xq(x_vec_in[i]), 
                .yq(ln_out_i)
            );

            // 2. Intra-lane Shifter (Math Reflector)
            always_comb begin
                case (shift_mode)
                    2'b01:   shift_out_i = -ln_out_i;
                    2'b10:   shift_out_i = -(ln_out_i >>> 1);
                    default: shift_out_i = ln_out_i;
                endcase
            end

            // 3. Intra-lane Routing MUX
            assign exp_in_i = exp_in_sel ? shift_out_i : x_vec_in[i];

            // 4. Pipeline Registers
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) exp_in_i_pipe <= '0;
                else        exp_in_i_pipe <= exp_in_i;
            end

            // 5. Exponential 
            tr_exp_alu #(
                .WIDTH(8), 
                .FRAC_W(4), 
                .LUT_IDX_W(3)
            ) u_exp_i (
                .x(exp_in_i_pipe), 
                .a_idx(shared_idx[i]),  // Hooked to Shared ROM[i]
                .mantisa(y_man_vec_out[i]),
                .is_zero(z_i)
            );

            // Apply the e^0 bypass 
            assign y_ea_vec_out[i] = z_i ? 8'd255 : shared_ea_raw[i];
        end
    endgenerate

endmodule