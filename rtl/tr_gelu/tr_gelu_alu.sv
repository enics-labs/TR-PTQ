/*
 * @module   tr_gelu (Two-Pass Architecture)
 * @brief    Taylor-Region GELU Activation Function.
 * @details  Mode 0: Computes Reciprocal Denominator (inv_S).
 *           Mode 1: Computes Final GELU Product using x_in and inv_S_in.
 */
`timescale 1ns/1ps

module tr_gelu_alu #(
    parameter int W = 8  // Q4.4 format (1 sign, 3 int, 4 frac)
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 valid_in,
    input  logic                 mode,       // 0: Pass 1 (inv_S), 1: Pass 2 (GELU Product)
    input  logic signed [W-1:0]  x_in,
    input  logic [7:0]           inv_s_in,   // Only used in Mode 1

    output logic                 valid_out,
    output logic signed [W-1:0]  gelu_out
);
    localparam int FRAC_W = 4;

    // --- Internal Signal Declarations ---
    logic signed [W-1:0] alpha_g, x_scaled_comb, x_s1, z_s1;
    logic [7:0]          inv_s1;
    logic                mode_s1, valid_s1;

    logic signed [W-1:0] u_neg_abs, x_s2, z_s2;
    logic [7:0]          inv_s2;
    logic                mode_s2, valid_s2;

    logic [7:0]          e_a, e_man, E_s3;
    logic                is_zero;
    logic signed [W-1:0] x_s3, z_s3;
    logic [7:0]          inv_s3;
    logic                mode_s3, valid_s3;

    // ========================================================================
    // STAGE 1: Scaling & Input Capture
    // ========================================================================
    always_comb begin
        logic [7:0] abs_z;
        logic signed [15:0] x_mult;
        abs_z = (x_in[W-1]) ? -x_in : x_in;
        
        case (abs_z[6:4]) 
            3'b000:  alpha_g = 8'h1B; // ~1.702
            3'b001:  alpha_g = 8'h1A;
            3'b010:  alpha_g = 8'h19;
            default: alpha_g = 8'h18;
        endcase
        
        x_mult = x_in * $signed({1'b0, alpha_g});
        
        if ($signed(x_mult) > 16'sd2032)       x_scaled_comb = 8'sd127;
        else if ($signed(x_mult) < -16'sd2048) x_scaled_comb = -8'sd128;
        else                                   x_scaled_comb = x_mult[11:4]; 
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_s1 <= '0; z_s1 <= '0; inv_s1 <= '0; 
            mode_s1 <= 1'b0; valid_s1 <= 1'b0;
        end else begin
            x_s1     <= x_scaled_comb;
            z_s1     <= x_in;
            inv_s1   <= inv_s_in; // Capture Pass 2 auxiliary input
            mode_s1  <= mode;
            valid_s1 <= valid_in;
        end
    end

    // ========================================================================
    // STAGE 2: Numerical Stabilization (u = -|x|)
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            u_neg_abs <= '0; x_s2 <= '0; z_s2 <= '0; inv_s2 <= '0;
            mode_s2 <= 1'b0; valid_s2 <= 1'b0;
        end else begin
            u_neg_abs <= (x_s1 > 0) ? -x_s1 : x_s1;
            x_s2      <= x_s1;
            z_s2      <= z_s1;
            inv_s2    <= inv_s1;
            mode_s2   <= mode_s1;
            valid_s2  <= valid_s1;
        end
    end

    // ========================================================================
    // STAGE 3: Single Exponential (E = e^-|x|)
    // ========================================================================
    tr_exp #(.FRAC_W(FRAC_W), .ITER(0)) u_exp (
        .x(u_neg_abs), .e_a(e_a), .mantisa(e_man), .is_zero(is_zero)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            E_s3 <= '0; x_s3 <= '0; z_s3 <= '0; inv_s3 <= '0;
            mode_s3 <= 1'b0; valid_s3 <= 1'b0;
        end else begin
            E_s3     <= is_zero ? 8'd0 : e_a; 
            x_s3     <= x_s2;
            z_s3     <= z_s2;
            inv_s3   <= inv_s2;
            mode_s3  <= mode_s2;
            valid_s3 <= valid_s2;
        end
    end

    // ========================================================================
    // PATH A: MODE 0 (Reciprocal Denominator Generation)
    // ========================================================================
    logic [7:0] inv_S_q4; 
    wire [8:0] sum_S = 9'd256 + E_s3; 
    
    tr_reciprocal #(.IN_WIDTH(9), .OUT_WIDTH(8), .IN_FRAC(4), .OUT_FRAC(4), .ITER(1)) 
    u_recip (.clk(clk), .rst_n(rst_n), .xq(sum_S >> 4), .yq(inv_S_q4));

    // Valid Shift Register specifically for Mode 0 Reciprocal Latency
    localparam int SYNC = 4;
    logic v_pipe_m0 [SYNC];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(int i=0; i<SYNC; i++) v_pipe_m0[i] <= 0;
        end else begin
            v_pipe_m0[0] <= valid_s3 && (mode_s3 == 1'b0); // Only propagate if Mode 0
            for(int i=1; i<SYNC; i++) begin
                v_pipe_m0[i] <= v_pipe_m0[i-1];
            end
        end
    end

    wire valid_m0_out = v_pipe_m0[SYNC-1];

    // ========================================================================
    // PATH B: MODE 1 (Symmetry Selection & Final Product)
    // ========================================================================
    logic signed [W-1:0] gelu_m1_out_reg;
    logic                valid_m1_out_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gelu_m1_out_reg  <= '0;
            valid_m1_out_reg <= 1'b0;
        end else begin
            logic [7:0]  sigmoid_q8;
            logic [15:0] sig_neg_mult;
            logic signed [15:0] gelu_prod;

            if (x_s3 > 0) begin
                sigmoid_q8 = inv_s3; 
            end else begin
                sig_neg_mult = (E_s3 * inv_s3);
                sigmoid_q8   = sig_neg_mult[15:8];
            end

            gelu_prod = z_s3 * $signed({1'b0, sigmoid_q8});
            gelu_m1_out_reg <= gelu_prod[15:8];
            
            // Output is valid immediately at Stage 4 for Mode 1
            valid_m1_out_reg <= valid_s3 && (mode_s3 == 1'b1);
        end
    end

    // ========================================================================
    // FINAL OUTPUT MUX
    // ========================================================================
    always_comb begin
        valid_out = valid_m0_out | valid_m1_out_reg;
        
        if (valid_m0_out) begin
            // Mode 0 outputs the unsigned Q0.8 reciprocal result
            gelu_out = $signed(inv_S_q4); 
        end else begin
            // Mode 1 outputs the final signed Q4.4 GELU
            gelu_out = gelu_m1_out_reg;   
        end
    end

endmodule