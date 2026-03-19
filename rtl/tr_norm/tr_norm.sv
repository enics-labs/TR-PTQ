module tr_norm #(
    parameter int N = 5,          // Arbitrary Vector size
    parameter int W = 8,          // I/O Width (Q4.4)
    parameter int ACC_W = 20      // Accumulator width to prevent overflow
)(
    input  logic                 clk,
    input  logic                 rst_n,
    
    input  logic                 valid_in,
    input  logic                 last_in,  // Pulses high on the last element of the vector
    input  logic signed [W-1:0]  x_in,

    output logic                 valid_out,
    output logic signed [W-1:0]  mean_out,
    output logic signed [ACC_W-1:0] var_out, // Kept wide for the 12-bit TR-Ln stage
    output logic [11:0]          inv_std_dev_out
);

    // ========================================================================
    // CONSTANT RECIPROCAL (1/N Computed at compile time)
    // ========================================================================
    localparam int RECIP_FRAC = 16;
    localparam signed [31:0] RECIP_N = (1 << RECIP_FRAC) / N;

    // ========================================================================
    // PIPELINE STAGE 1: Square the input
    // ========================================================================
    logic signed [W-1:0]     x_reg;
    logic signed [(W*2)-1:0] x_sq_reg;
    logic                    valid_s1;
    logic                    last_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg    <= '0;
            x_sq_reg <= '0;
            valid_s1 <= 1'b0;
            last_s1  <= 1'b0;
        end else if (valid_in) begin
            x_reg    <= x_in;
            x_sq_reg <= x_in * x_in; // Q4.4 * Q4.4 = Q8.8
            valid_s1 <= 1'b1;
            last_s1  <= last_in;
        end else begin
            valid_s1 <= 1'b0;
            last_s1  <= 1'b0;
        end
    end

    // ========================================================================
    // PIPELINE STAGE 2: Accumulate E[X] and E[X^2]
    // ========================================================================
    logic signed [ACC_W-1:0] sum_x;
    logic signed [ACC_W-1:0] sum_x_sq;
    logic                    trigger_calc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_x        <= '0;
            sum_x_sq     <= '0;
            trigger_calc <= 1'b0;
        end else begin
            trigger_calc <= last_s1;
            
            if (valid_s1) begin
                // If this is the first element of a new vector (the cycle after 'last'), reset sums
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

    // ========================================================================
    // PIPELINE STAGE 3: Final Variance Calculation: Var(X) = E[X^2] - (E[X])^2
    // ========================================================================
    // We need wider logic to hold the multiplication before shifting
    logic signed [ACC_W+31:0] mean_val_full;
    logic signed [ACC_W+31:0] mean_of_sq_full;

    logic signed [ACC_W-1:0] mean_val;
    logic signed [ACC_W+3:0] mean_val_q8;
    logic signed [ACC_W-1:0] mean_sq;
    logic signed [ACC_W-1:0] mean_of_sq;
    logic signed [ACC_W-1:0] var_comb;

    logic signed [63:0] mean_sq_full;
    
    always_comb begin
        // Multiply by the reciprocal constant
        mean_val_full   = sum_x * RECIP_N;
        mean_of_sq_full = sum_x_sq * RECIP_N;

        // Qx.4 output for the standard mean_out port
        mean_val = (mean_val_full + (1 << (RECIP_FRAC - 1))) >>> RECIP_FRAC;    
        
        // Qx.8 internal mean for precision squaring (shift by RECIP_FRAC - 4 = 12)
        mean_val_q8 = (mean_val_full + (1 << (RECIP_FRAC - 5))) >>> (RECIP_FRAC - 4);
        
        mean_of_sq = (mean_of_sq_full + (1 << (RECIP_FRAC - 1))) >>> RECIP_FRAC;    
        
        // Square the Qx.8 mean safely
        mean_sq_full = mean_val_q8 * mean_val_q8;
        mean_sq = (mean_sq_full + (1 << 7)) >>> 8; // Shift back to Qx.8
        
        var_comb = mean_of_sq - mean_sq;   
    end

    // ========================================================================
    // PIPELINE STAGE 4: Inverse Square Root (1 / sqrt(Var + Epsilon))
    // ========================================================================
    logic [ACC_W-1:0] safe_var;
    logic [11:0]      inv_std_dev_math;
    logic [11:0]      inv_std_dev_comb;

    tr_reciprocal #(
        .IN_WIDTH  (ACC_W),  // 20-bit variance
        .OUT_WIDTH (12),     // 12-bit multiplier output
        .IN_FRAC   (8),
        .OUT_FRAC  (8),
        .INV_SQRT  (1),      // <--- MAGIC SWITCH: Now it computes 1/sqrt(x) !
        .ITER      (2)       // 2nd-order accuracy for LayerNorm
    ) u_isqrt (
        .clk   (clk),
        .rst_n (rst_n),
        .xq    (var_comb),
        .yq    (inv_std_dev_math)
    );

    always_comb begin
        if      (var_comb <= 0) inv_std_dev_comb = 12'd0;    // Squelch noise to 0!
        else if (var_comb == 1) inv_std_dev_comb = 12'd4095; // ~16.00 
        else if (var_comb == 2) inv_std_dev_comb = 12'd2896; // ~11.31
        else if (var_comb == 3) inv_std_dev_comb = 12'd2364; // ~9.23
        else                    inv_std_dev_comb = inv_std_dev_math;
    end

    // ========================================================================
    // OUTPUT REGISTERS
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mean_out        <= '0;
            var_out         <= '0;
            inv_std_dev_out <= '0;
            valid_out       <= 1'b0;
        end else begin
            valid_out <= trigger_calc;
            if (trigger_calc) begin
                mean_out        <= mean_val[W-1:0];
                var_out         <= var_comb; 
                inv_std_dev_out <= inv_std_dev_comb;
            end
        end
    end

endmodule