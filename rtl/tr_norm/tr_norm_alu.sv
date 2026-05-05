/*
 * @module   tr_norm_multipass
 * @brief    Two-Pass Taylor-Region LayerNorm.
 * @details  Mode 0: Computes Mean and Inverse Std Dev (Stats Pass).
 *           Mode 1: Computes Affine Normalized Output (Norm Pass).
 */
`timescale 1ns/1ps

module tr_norm_alu #(
    parameter int N = 5,
    parameter int W = 8,
    parameter int ACC_W = 20,
    parameter int ISQRT_LATENCY = 6
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

    // Stream Splitters
    wire valid_m0 = valid_in & ~mode;
    wire valid_m1 = valid_in & mode;

    // ========================================================================
    // PATH A: MODE 0 (Pass 1 - Statistics Generation)
    // ========================================================================
    localparam int RECIP_FRAC = 16;
    localparam signed [31:0] RECIP_N = (1 << RECIP_FRAC) / N;

    // --- S1: Square Input ---
    logic signed [W-1:0]     x_reg;
    logic signed [(W*2)-1:0] x_sq_reg;
    logic                    valid_s1_m0;
    logic                    last_s1_m0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg <= '0; x_sq_reg <= '0;
            valid_s1_m0 <= 1'b0; last_s1_m0 <= 1'b0;
        end else if (valid_m0) begin
            x_reg <= x_in;
            x_sq_reg <= x_in * x_in; 
            valid_s1_m0 <= 1'b1;
            last_s1_m0 <= last_in;
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
                if (trigger_calc) begin // Reset on new vector
                    sum_x    <= $signed(x_reg);
                    sum_x_sq <= $signed(x_sq_reg);
                end else begin
                    sum_x    <= sum_x + $signed(x_reg);
                    sum_x_sq <= sum_x_sq + $signed(x_sq_reg);
                end
            end
        end
    end

    // --- S3: Variance Math ---
    logic signed [ACC_W-1:0] mean_val;
    logic signed [ACC_W-1:0] var_comb;
    
    always_comb begin
        logic signed [ACC_W+31:0] mean_val_full   = sum_x * RECIP_N;
        logic signed [ACC_W+31:0] mean_of_sq_full = sum_x_sq * RECIP_N;
        
        logic signed [ACC_W+3:0] mean_val_q8;
        logic signed [ACC_W-1:0] mean_sq, mean_of_sq;
        logic signed [63:0]      mean_sq_full;

        mean_val    = (mean_val_full + (1 << (RECIP_FRAC - 1))) >>> RECIP_FRAC;    
        mean_val_q8 = (mean_val_full + (1 << (RECIP_FRAC - 5))) >>> (RECIP_FRAC - 4);
        mean_of_sq  = (mean_of_sq_full + (1 << (RECIP_FRAC - 1))) >>> RECIP_FRAC;    
        
        mean_sq_full = mean_val_q8 * mean_val_q8;
        mean_sq      = (mean_sq_full + (1 << 7)) >>> 8; 
        
        var_comb     = mean_of_sq - mean_sq;   
    end

    // --- S4: Inverse Square Root ---
    logic [11:0] inv_std_dev_math, inv_std_dev_comb;

    tr_reciprocal #(
        .IN_WIDTH(ACC_W), .OUT_WIDTH(12), .IN_FRAC(8), .OUT_FRAC(8), .INV_SQRT(1), .ITER(2)
    ) u_isqrt (
        .clk(clk), .rst_n(rst_n), .xq(var_comb), .yq(inv_std_dev_math)
    );

    // --- S4 Delay Pipeline ---
    logic signed [W-1:0]     mean_pipe  [ISQRT_LATENCY];
    logic signed [ACC_W-1:0] var_pipe   [ISQRT_LATENCY];
    logic                    valid_pipe [ISQRT_LATENCY];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(int i=0; i<ISQRT_LATENCY; i++) begin
                mean_pipe[i] <= '0; var_pipe[i] <= '0; valid_pipe[i] <= 1'b0;
            end
        end else begin
            mean_pipe[0]  <= mean_val[W-1:0];
            var_pipe[0]   <= var_comb;
            valid_pipe[0] <= trigger_calc;
            for(int i=1; i<ISQRT_LATENCY; i++) begin
                mean_pipe[i]  <= mean_pipe[i-1];
                var_pipe[i]   <= var_pipe[i-1];
                valid_pipe[i] <= valid_pipe[i-1];
            end
        end
    end

    always_comb begin
        if      (var_pipe[ISQRT_LATENCY-1] <= 0) inv_std_dev_comb = 12'd0;    
        else if (var_pipe[ISQRT_LATENCY-1] == 1) inv_std_dev_comb = 12'd4095; 
        else if (var_pipe[ISQRT_LATENCY-1] == 2) inv_std_dev_comb = 12'd2896; 
        else if (var_pipe[ISQRT_LATENCY-1] == 3) inv_std_dev_comb = 12'd2364; 
        else                                     inv_std_dev_comb = inv_std_dev_math;
    end

    // ========================================================================
    // PATH B: MODE 1 (Pass 2 - Normalization & Affine)
    // ========================================================================

    // --- M1_S1: Fetch & Center ---
    logic signed [W:0]   x_centered_s1; 
    logic                valid_s1_m1, last_s1_m1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1_m1 <= 1'b0; last_s1_m1 <= 1'b0;
            x_centered_s1 <= '0;
        end else begin
            valid_s1_m1 <= valid_m1;
            last_s1_m1  <= last_in & mode;
            if (valid_m1) begin
                x_centered_s1 <= $signed({x_in[W-1], x_in}) - $signed({mean_in[W-1], mean_in});
            end
        end
    end

    // --- M1_S2: Scale by Inverse Std Dev ---
    logic signed [W-1:0] x_scaled_s2;
    logic                valid_s2_m1, last_s2_m1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2_m1 <= 1'b0; last_s2_m1 <= 1'b0; x_scaled_s2 <= '0;
        end else begin
            valid_s2_m1 <= valid_s1_m1; 
            last_s2_m1  <= last_s1_m1;
            if (valid_s1_m1) begin
                logic signed [24:0] full_mult = x_centered_s1 * $signed({1'b0, inv_std_in});
                logic signed [24:0] shifted_val = full_mult >>> 8; // Q4.4 * Q4.8 -> Shift 8 -> Q4.4
                
                if (shifted_val > 127)       x_scaled_s2 <= 127;
                else if (shifted_val < -128) x_scaled_s2 <= -128;
                else                         x_scaled_s2 <= shifted_val[7:0]; 
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
                logic signed [16:0] affine_mult = x_scaled_s2 * gamma;
                logic signed [17:0] affine_add  = affine_mult + (beta <<< 4);
                logic signed [17:0] shifted_y   = affine_add >>> 4;
                
                if (shifted_y > 127)       y_m1_out <= 127;
                else if (shifted_y < -128) y_m1_out <= -128;
                else                       y_m1_out <= shifted_y[7:0];
            end
        end
    end

    // ========================================================================
    // FINAL OUTPUT ROUTING
    // ========================================================================
    always_comb begin
        // The output bus dynamically switches between Mode 0 Stats and Mode 1 Data
        valid_out       = valid_pipe[ISQRT_LATENCY-1] | valid_s3_m1;
        last_out        = last_s3_m1;
        
        mean_out        = mean_pipe[ISQRT_LATENCY-1];
        inv_std_dev_out = inv_std_dev_comb;
        
        y_out           = y_m1_out;
    end

endmodule