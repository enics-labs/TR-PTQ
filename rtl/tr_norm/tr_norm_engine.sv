module tr_norm_engine #(
    parameter int N = 8,          // Vector size
    parameter int W = 8,          // I/O Width (Q4.4)
    parameter int ACC_W = 20,     // Accumulator width
    parameter int FIFO_DEPTH = 32 // Must be > N + latency
)(
    input  logic                 clk,
    input  logic                 rst_n,
    
    // Input Stream
    input  logic                 valid_in,
    input  logic                 last_in,
    input  logic signed [W-1:0]  x_in,
    
    // Affine Weights (Learned Parameters)
    input  logic signed [W-1:0]  gamma, // Scale (Q4.4)
    input  logic signed [W-1:0]  beta,  // Bias  (Q4.4)

    // Output Stream
    output logic                 valid_out,
    output logic                 last_out,
    output logic signed [W-1:0]  y_out
);

    // ========================================================================
    // 1. TR-Norm Core (Statistics Generator)
    // ========================================================================
    logic                 stats_valid;
    logic signed [W-1:0]  stats_mean;
    logic [11:0]          stats_inv_std_dev;

    tr_norm #(
        .N(N), .W(W), .ACC_W(ACC_W)
    ) u_tr_norm (
        .clk             (clk),
        .rst_n           (rst_n),
        .valid_in        (valid_in),
        .last_in         (last_in),
        .x_in            (x_in),
        .valid_out       (stats_valid),
        .mean_out        (stats_mean),
        .var_out         (), 
        .inv_std_dev_out (stats_inv_std_dev)
    );

    // ========================================================================
    // 2. The Data FIFO (Circular Buffer)
    // ========================================================================
    logic signed [W-1:0] x_fifo [0:FIFO_DEPTH-1];
    logic [$clog2(FIFO_DEPTH)-1:0] wr_ptr;
    logic [$clog2(FIFO_DEPTH)-1:0] rd_ptr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
        end else if (valid_in) begin
            x_fifo[wr_ptr] <= x_in;
            wr_ptr <= (wr_ptr == FIFO_DEPTH-1) ? '0 : wr_ptr + 1;
        end
    end

    // ========================================================================
    // 3. Affine Weights Capture (Shift Register)
    // ========================================================================
    // Captures gamma/beta at the end of the input vector and delays them 
    // to align perfectly with the stats_valid pulse from tr_norm.
    logic signed [W-1:0] gamma_d1, beta_d1;
    logic signed [W-1:0] gamma_d2, beta_d2;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gamma_d1 <= '0; beta_d1 <= '0;
            gamma_d2 <= '0; beta_d2 <= '0;
        end else begin
            if (valid_in && last_in) begin
                gamma_d1 <= gamma;
                beta_d1  <= beta;
            end
            gamma_d2 <= gamma_d1;
            beta_d2  <= beta_d1;
        end
    end

    // ========================================================================
    // 4. The Stats FIFO
    // ========================================================================
    // Buffers the parameters so back-to-back vectors never overwrite each other
    logic signed [W-1:0] stats_f_mean [0:3];
    logic [11:0]         stats_f_inv  [0:3];
    logic signed [W-1:0] stats_f_g    [0:3];
    logic signed [W-1:0] stats_f_b    [0:3];
    logic [1:0]          stats_wr_ptr;
    logic [1:0]          stats_rd_ptr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stats_wr_ptr <= '0;
        end else if (stats_valid) begin
            stats_f_mean[stats_wr_ptr] <= stats_mean;
            stats_f_inv[stats_wr_ptr]  <= stats_inv_std_dev;
            stats_f_g[stats_wr_ptr]    <= gamma_d2;
            stats_f_b[stats_wr_ptr]    <= beta_d2;
            stats_wr_ptr <= stats_wr_ptr + 1;
        end
    end

    // ========================================================================
    // 5. Output Stream Controller
    // ========================================================================
    logic pop_active;
    logic [$clog2(N+1)-1:0] pop_count;
    logic stats_empty;

    assign stats_empty = (stats_wr_ptr == stats_rd_ptr);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pop_active   <= 1'b0;
            pop_count    <= '0;
            rd_ptr       <= '0;
            stats_rd_ptr <= '0;
        end else begin
            if (pop_active) begin
                rd_ptr <= (rd_ptr == FIFO_DEPTH-1) ? '0 : rd_ptr + 1;
                pop_count <= pop_count - 1;
                
                if (pop_count == 1) begin
                    stats_rd_ptr <= stats_rd_ptr + 1; // Advance to next vector's stats
                    if (!stats_empty) begin
                        pop_active <= 1'b1;
                        pop_count  <= N;
                    end else begin
                        pop_active <= 1'b0;
                    end
                end
            end else begin
                if (!stats_empty) begin
                    pop_active <= 1'b1;
                    pop_count  <= N;
                end
            end
        end
    end

    // ========================================================================
    // 6. Affine Transformation Pipeline
    // ========================================================================
    // STAGE 1: Fetch & Center
    logic signed [W-1:0] x_raw_s1;
    logic signed [W:0]   x_centered_s1; 
    logic [11:0]         s1_inv;
    logic signed [W-1:0] s1_gamma, s1_beta;
    logic                valid_s1, last_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1 <= 1'b0; last_s1 <= 1'b0;
            x_raw_s1 <= '0; x_centered_s1 <= '0;
            s1_inv <= '0; s1_gamma <= '0; s1_beta <= '0;
        end else begin
            valid_s1 <= pop_active;
            last_s1  <= pop_active && (pop_count == 1);
            
            if (pop_active) begin
                x_raw_s1 <= x_fifo[rd_ptr];
                x_centered_s1 <= $signed({x_fifo[rd_ptr][W-1], x_fifo[rd_ptr]}) - 
                                 $signed({stats_f_mean[stats_rd_ptr][W-1], stats_f_mean[stats_rd_ptr]});
                
                // Pipeline the stats alongside the data!
                s1_inv   <= stats_f_inv[stats_rd_ptr];
                s1_gamma <= stats_f_g[stats_rd_ptr];
                s1_beta  <= stats_f_b[stats_rd_ptr];
            end
        end
    end

    // STAGE 2: Scale by Inverse Standard Deviation 
    logic signed [W-1:0] x_scaled_s2;
    logic signed [W-1:0] s2_gamma, s2_beta;
    logic                valid_s2, last_s2;
    logic signed [24:0]  full_mult;
    logic signed [24:0]  shifted_val;

    always_comb begin
        full_mult = x_centered_s1 * $signed({1'b0, s1_inv});
        shifted_val = full_mult >>> 8;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s2 <= 1'b0; last_s2 <= 1'b0; 
            x_scaled_s2 <= '0; s2_gamma <= '0; s2_beta <= '0;
        end else begin
            valid_s2 <= valid_s1; 
            last_s2  <= last_s1;
            
            if (valid_s1) begin
                // Pipeline the weights to Stage 3!
                s2_gamma <= s1_gamma;
                s2_beta  <= s1_beta;
                
                if (shifted_val > 127)       x_scaled_s2 <= 127;
                else if (shifted_val < -128) x_scaled_s2 <= -128;
                else                         x_scaled_s2 <= shifted_val[7:0]; 
            end
        end
    end

    // STAGE 3: Affine Transform 
    logic signed [16:0] affine_mult;
    logic signed [17:0] affine_add;
    logic signed [17:0] shifted_y;

    always_comb begin
        affine_mult = x_scaled_s2 * s2_gamma;
        affine_add  = affine_mult + (s2_beta <<< 4);
        shifted_y   = affine_add >>> 4;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0; last_out <= 1'b0; y_out <= '0;
        end else begin
            valid_out <= valid_s2; 
            last_out  <= last_s2;
            
            if (valid_s2) begin
                if (shifted_y > 127)       y_out <= 127;
                else if (shifted_y < -128) y_out <= -128;
                else                       y_out <= shifted_y[7:0];
            end
        end
    end

endmodule