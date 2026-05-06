/**
 * @module   tr_norm_alu
 * @brief    Two-Pass Taylor-Region LayerNorm (Fully Pipelined)
 * @details  Mode 0: Computes Mean and Inverse Std Dev (Stats Pass).
 *           Mode 1: Computes Affine Normalized Output (Norm Pass).
 */
`timescale 1ns/1ps

module tr_norm_alu #(
    parameter int N = 8,              // Vector Size
    parameter int W = 8,              // I/O Width
    parameter int FRAC_W = 4,         // Fractional bits (e.g., 4 for Q4.4)
    parameter int ACC_W = 20,         // Accumulator Width
    parameter int ISQRT_LATENCY = 6   // Internal Pipeline Latency for Pass 1
)(
    input  logic                 clk,
    input  logic                 rst_n,
    
    // Core Stream Inputs
    input  logic                 valid_in,
    input  logic                 last_in,
    input  logic                 mode,       // 0: Pass 1 (Stats), 1: Pass 2 (Norm)
    input  logic signed [W-1:0]  x_in,
    
    // Auxiliary Inputs for Pass 2
    input  logic signed [W-1:0]  mean_in,
    input  logic [11:0]          inv_std_in,
    input  logic signed [W-1:0]  gamma,
    input  logic signed [W-1:0]  beta,

    // Unified Outputs
    output logic                 valid_out,
    output logic                 last_out,
    output logic signed [W-1:0]  mean_out,
    output logic [11:0]          inv_std_dev_out,
    output logic signed [W-1:0]  y_out
);

    // ========================================================================
    // DYNAMIC LIMITS & STREAM ROUTING
    // ========================================================================
    localparam signed [2*W-1:0] MAX_VAL =  (1 << (W-1)) - 1;
    localparam signed [2*W-1:0] MIN_VAL = -(1 << (W-1));

    wire valid_m0 = valid_in & ~mode;
    wire valid_m1 = valid_in & mode;

    // ========================================================================
    // PATH A: MODE 0 (Pass 1 - Statistics Generation)
    // ========================================================================
    localparam int RECIP_FRAC = 16;
    localparam signed [31:0] RECIP_N = (1 << RECIP_FRAC) / N;
    
    localparam int MEAN_SQ_SHIFT = RECIP_FRAC + FRAC_W - 8;
    localparam int VAR_SHIFT     = RECIP_FRAC + (2 * FRAC_W) - 8;

    // --- S1: Square Input ---
    logic signed [W-1:0]     x_reg;
    logic signed [(W*2)-1:0] x_sq_reg;
    logic                    valid_s1_m0, last_s1_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg <= '0; x_sq_reg <= '0;
            valid_s1_m0 <= 1'b0; last_s1_m0 <= 1'b0;
        end else if (valid_m0) begin
            x_reg <= x_in;
            x_sq_reg <= x_in * x_in; 
            valid_s1_m0 <= 1'b1; last_s1_m0 <= last_in;
        end else begin
            valid_s1_m0 <= 1'b0; last_s1_m0 <= 1'b0;
        end
    end

    // --- S2: Accumulate ---
    logic signed [ACC_W-1:0] sum_x, sum_x_sq;
    logic                    trigger_calc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_x <= '0; sum_x_sq <= '0; trigger_calc <= 1'b0;
        end else begin
            trigger_calc <= last_s1_m0;
            if (valid_s1_m0) begin
                if (trigger_calc) begin 
                    sum_x    <= $signed(x_reg);
                    sum_x_sq <= $signed(x_sq_reg);
                end else begin
                    sum_x    <= sum_x + $signed(x_reg);
                    sum_x_sq <= sum_x_sq + $signed(x_sq_reg);
                end
            end
        end
    end

    // --- S3: Scale by 1/N ---
    logic signed [ACC_W+31:0] mean_val_full_s3;
    logic signed [ACC_W+31:0] mean_of_sq_full_s3;
    logic                     valid_s3_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mean_val_full_s3   <= '0;
            mean_of_sq_full_s3 <= '0;
            valid_s3_m0        <= 1'b0;
        end else begin
            mean_val_full_s3   <= sum_x * RECIP_N;
            mean_of_sq_full_s3 <= sum_x_sq * RECIP_N;
            valid_s3_m0        <= trigger_calc;
        end
    end

    // --- S4: Square & Subtract ---
    logic signed [W-1:0]     mean_reg_s4;
    logic signed [ACC_W-1:0] var_reg_s4;
    logic                    valid_s4_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mean_reg_s4 <= '0; var_reg_s4 <= '0; valid_s4_m0 <= 1'b0;
        end else begin
            logic signed [ACC_W-1:0] mean_val;
            logic signed [ACC_W+3:0] mean_val_q8;
            logic signed [ACC_W-1:0] mean_sq, mean_of_sq;
            logic signed [63:0]      mean_sq_full;

            mean_val    = (mean_val_full_s3 + (1 << (RECIP_FRAC - 1))) >>> RECIP_FRAC;    
            mean_val_q8 = (mean_val_full_s3 + (1 << (MEAN_SQ_SHIFT - 1))) >>> MEAN_SQ_SHIFT;
            mean_of_sq  = (mean_of_sq_full_s3 + (1 << (VAR_SHIFT - 1))) >>> VAR_SHIFT;    
            
            mean_sq_full = mean_val_q8 * mean_val_q8;
            mean_sq      = (mean_sq_full + (1 << 7)) >>> 8; 
            
            mean_reg_s4 <= mean_val[W-1:0];
            var_reg_s4  <= mean_of_sq - mean_sq;
            valid_s4_m0 <= valid_s3_m0;
        end
    end

    // ========================================================================
    // S5 to S9: Fully Pipelined Inverse Square Root 
    // ========================================================================

    // --- S5: Logarithm Stage ---
    logic signed [11:0] ln_x_comb; 
    
    tr_ln #(
        .WIDTH(ACC_W), .BITS(8), .OUT_WIDTH(12)
    ) u_ln (
        .xq(var_reg_s4), .yq(ln_x_comb)
    );

    logic signed [11:0]      ln_x_s5;
    logic signed [W-1:0]     mean_s5;
    logic signed [ACC_W-1:0] var_s5;
    logic                    valid_s5_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ln_x_s5 <= '0; mean_s5 <= '0; var_s5 <= '0; valid_s5_m0 <= 1'b0;
        end else begin
            ln_x_s5     <= ln_x_comb;
            mean_s5     <= mean_reg_s4;
            var_s5      <= var_reg_s4;
            valid_s5_m0 <= valid_s4_m0;
        end
    end

    // --- S6: Math Selector (Negate & Shift for 1/sqrt(x)) ---
    logic signed [11:0]      neg_ln_x_s6;
    logic signed [W-1:0]     mean_s6;
    logic signed [ACC_W-1:0] var_s6;
    logic                    valid_s6_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            neg_ln_x_s6 <= '0; mean_s6 <= '0; var_s6 <= '0; valid_s6_m0 <= 1'b0;
        end else begin
            neg_ln_x_s6 <= -(ln_x_s5 >>> 1); // LayerNorm mode x^(-0.5)
            mean_s6     <= mean_s5;
            var_s6      <= var_s5;
            valid_s6_m0 <= valid_s5_m0;
        end
    end

    // --- S7: Exponential Stage ---
    logic [11:0] e_a_comb, mantisa_comb;
    
    tr_exp #(
        .WIDTH(12), .FRAC_W(8), .LUT_IDX_W(4), .ITER(2)
    ) u_exp (
        .x(neg_ln_x_s6), .e_a(e_a_comb), .mantisa(mantisa_comb), .is_zero()
    );

    logic [11:0]             e_a_s7, mantisa_s7;
    logic signed [W-1:0]     mean_s7;
    logic signed [ACC_W-1:0] var_s7;
    logic                    valid_s7_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            e_a_s7 <= '0; mantisa_s7 <= '0; 
            mean_s7 <= '0; var_s7 <= '0; valid_s7_m0 <= 1'b0;
        end else begin
            e_a_s7      <= e_a_comb;
            mantisa_s7  <= mantisa_comb;
            mean_s7     <= mean_s6;
            var_s7      <= var_s6;
            valid_s7_m0 <= valid_s6_m0;
        end
    end

    // --- S8: Final Assembly Multiplier ---
    logic [23:0]             mult_res_s8;
    logic signed [W-1:0]     mean_s8;
    logic signed [ACC_W-1:0] var_s8;
    logic                    valid_s8_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mult_res_s8 <= '0; mean_s8 <= '0; var_s8 <= '0; valid_s8_m0 <= 1'b0;
        end else begin
            mult_res_s8 <= e_a_s7 * mantisa_s7;
            mean_s8     <= mean_s7;
            var_s8      <= var_s7;
            valid_s8_m0 <= valid_s7_m0;
        end
    end

    // --- S9: Format Shift and Capture ---
    logic [11:0]             inv_std_dev_math_s9;
    logic signed [W-1:0]     mean_s9;
    logic signed [ACC_W-1:0] var_s9;
    logic                    valid_s9_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inv_std_dev_math_s9 <= '0; mean_s9 <= '0; var_s9 <= '0; valid_s9_m0 <= 1'b0;
        end else begin
            inv_std_dev_math_s9 <= mult_res_s8[19:8]; // Shift back to Q4.8
            mean_s9             <= mean_s8;
            var_s9              <= var_s8;
            valid_s9_m0         <= valid_s8_m0;
        end
    end

    // --- S10: Noise Gate (Combinational Output for Mode 0) ---
    logic [11:0] inv_std_dev_comb;
    always_comb begin
        if      (var_s9 <= 0) inv_std_dev_comb = 12'd0;    
        else if (var_s9 == 1) inv_std_dev_comb = 12'd4095; 
        else if (var_s9 == 2) inv_std_dev_comb = 12'd2896; 
        else if (var_s9 == 3) inv_std_dev_comb = 12'd2364; 
        else                  inv_std_dev_comb = inv_std_dev_math_s9;
    end

    // ========================================================================
    // PATH B: MODE 1 (Pass 2 - Normalization & Affine)
    // ========================================================================

    // --- M1_S1: Fetch, Center & Input Buffering ---
    logic signed [W:0]   x_centered_s1; 
    logic [11:0]         inv_std_s1;
    logic signed [W-1:0] gamma_s1, beta_s1;
    logic                valid_s1_m1, last_s1_m1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1_m1 <= 1'b0; last_s1_m1 <= 1'b0; x_centered_s1 <= '0;
            inv_std_s1 <= '0; gamma_s1 <= '0; beta_s1 <= '0;
        end else begin
            valid_s1_m1 <= valid_m1;
            last_s1_m1  <= last_in & mode;
            if (valid_m1) begin
                x_centered_s1 <= $signed({x_in[W-1], x_in}) - $signed({mean_in[W-1], mean_in});
                inv_std_s1    <= inv_std_in;  
                gamma_s1      <= gamma;
                beta_s1       <= beta;
            end
        end
    end

    // --- M1_S2: Scale by Inverse Std Dev ---
    logic signed [W-1:0] x_scaled_s2;
    logic signed [W-1:0] gamma_s2, beta_s2;
    logic                valid_s2_m1, last_s2_m1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2_m1 <= 1'b0; last_s2_m1 <= 1'b0; x_scaled_s2 <= '0;
            gamma_s2 <= '0; beta_s2 <= '0;
        end else begin
            valid_s2_m1 <= valid_s1_m1; 
            last_s2_m1  <= last_s1_m1;
            if (valid_s1_m1) begin
                logic signed [W+14:0] full_mult = x_centered_s1 * $signed({1'b0, inv_std_s1});
                logic signed [W+14:0] shifted_val = full_mult >>> 8; 
                
                gamma_s2 <= gamma_s1;
                beta_s2  <= beta_s1;
                
                if (shifted_val > MAX_VAL)       x_scaled_s2 <= MAX_VAL[W-1:0];
                else if (shifted_val < MIN_VAL)  x_scaled_s2 <= MIN_VAL[W-1:0];
                else                             x_scaled_s2 <= shifted_val[W-1:0]; 
            end
        end
    end

    // --- M1_S3: Affine Transform ---
    logic signed [W-1:0] y_m1_out;
    logic                valid_s3_m1, last_s3_m1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s3_m1 <= 1'b0; last_s3_m1 <= 1'b0; y_m1_out <= '0;
        end else begin
            valid_s3_m1 <= valid_s2_m1; 
            last_s3_m1  <= last_s2_m1;
            if (valid_s2_m1) begin
                logic signed [2*W:0] affine_mult = x_scaled_s2 * gamma_s2;
                logic signed [2*W:0] affine_add  = affine_mult + (beta_s2 <<< FRAC_W);
                logic signed [2*W:0] shifted_y   = affine_add >>> FRAC_W;
                
                if (shifted_y > MAX_VAL)       y_m1_out <= MAX_VAL[W-1:0];
                else if (shifted_y < MIN_VAL)  y_m1_out <= MIN_VAL[W-1:0];
                else                           y_m1_out <= shifted_y[W-1:0];
            end
        end
    end

    // ========================================================================
    // FINAL OUTPUT ROUTING
    // ========================================================================
    always_comb begin
        valid_out       = valid_s9_m0 | valid_s3_m1;
        last_out        = last_s3_m1;
        mean_out        = mean_s9;
        inv_std_dev_out = inv_std_dev_comb;
        y_out           = y_m1_out;
    end

endmodule