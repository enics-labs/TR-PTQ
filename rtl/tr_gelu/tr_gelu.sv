module tr_gelu #(
    parameter int W = 8  // Assuming Q4.4 format (1 sign, 3 int, 4 frac)
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 valid_in,
    input  logic signed [W-1:0]  x_in,

    // We will add the final outputs later
    output logic                 valid_out,
    output logic signed [W-1:0]  gelu_out
);

    localparam int FRAC_W = 4;

    // ========================================================================
    // STAGE 1: Alpha Refactor (Region-Dependent Scaling)
    // ========================================================================
    // Divides the positive half into segments to fetch a local alpha.
    // Because GELU has odd symmetry: GELU(-X) = -GELU(X), we use the absolute value to index the LUT.

    logic signed [W-1:0] abs_x;
    logic        [1:0]   region_idx; // 4 entries = 2-bit index 
    logic signed [W-1:0] alpha_g; 

    logic signed [W-1:0] x_scaled;
    logic signed [W-1:0] x_in_s1;
    logic                valid_s1;

    always_comb begin
        // 1. Get absolute value to exploit odd symmetry
        abs_x = (x_in[W-1]) ? -x_in : x_in;

        // 2. Extract the integer bits to determine the region
        // Assuming Q4.4, bits [6:4] are the integer magnitude. We cap it at 3 for the LUT.
        if (abs_x[6:5] != 2'b00) begin
            region_idx = 2'b11; // Saturate to the highest region
        end else begin
            region_idx = abs_x[5:4];
        end

        // 3. The 4-entry Alpha LUT
        // (Note: These are placeholder Q4.4 values for alpha_g. We need to 
        // plug in the offline-calculated MSE values here).
        case (region_idx)
            2'b00: alpha_g = 8'h1B; // ~1.702 in Q4.4
            2'b01: alpha_g = 8'h1A; 
            2'b10: alpha_g = 8'h19; 
            2'b11: alpha_g = 8'h18; 
        endcase
    end

    // --- Combinational multiplication and scaling ---
    logic signed [15:0] mult_res;
    logic signed [15:0] scaled_mult;
    logic signed [W-1:0] x_scaled_next;

    always_comb begin
        mult_res = x_in * $signed({1'b0, alpha_g});
        scaled_mult = mult_res >>> FRAC_W;
        
        // Clamp to 8-bit limits
        if (scaled_mult > 127)       x_scaled_next = 127;
        else if (scaled_mult < -128) x_scaled_next = -128;
        else                         x_scaled_next = scaled_mult[7:0];
    end

    // --- Pipeline Register 1 ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_scaled <= '0;
            x_in_s1  <= '0;
            valid_s1 <= 1'b0;
        end else begin
            valid_s1 <= valid_in;
            x_scaled <= x_scaled_next;
            x_in_s1  <= x_in;
        end
    end

    // ========================================================================
    // STAGE 2: Compression Block
    // ========================================================================
    // This calculates x_max = max(x, 0) to ensure the exponentials in the 
    // sigmoid denominator are always evaluated on non-positive numbers.
    // We need to generate two terms for the denominator: e^(-x_max) and e^(x - x_max).

    logic signed [W-1:0] x_max;
    logic signed [W-1:0] term1_in; // Input for e^(-x_max)
    logic signed [W-1:0] term2_in; // Input for e^(x - x_max)

    // The thesis notes this compresses to an int3 format to save power, but we will keep
    // it as W-1:0 here before passing it to the exponential LUT to maintain the variable width structure.
    logic signed [W-1:0] term1_reg;
    logic signed [W-1:0] term2_reg;
    logic signed [W-1:0] original_x_reg; // Need to delay the original x for the final multiply
    logic                valid_s2;

    always_comb begin
        // max(x, 0) logic 
        x_max = (x_scaled > 0) ? x_scaled : '0;
        
        // Calculate the two non-positive inputs for the exponential generators
        term1_in = -x_max;
        term2_in = x_scaled - x_max;
    end

    // Pipeline Register 2
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            term1_reg <= '0;
            term2_reg <= '0;
            original_x_reg <= '0;
            valid_s2  <= 1'b0;
        end else begin
            valid_s2       <= valid_s1;
            term1_reg      <= term1_in;
            term2_reg      <= term2_in;
            original_x_reg <= x_in_s1; // Propagate the scaled x
        end
    end

    // ========================================================================
    // STAGE 3: Sum Buffer (Exponentials & Accumulation)
    // ========================================================================
    // Instantiate two TR-exp blocks to calculate the numerator and the 
    // two terms of the denominator.

    localparam int STAGE3_ITER = 0;

    // Wires for Term 1: e^(-x_max)
    logic [7:0] e_a_1;
    logic [7:0] mantisa_1;
    logic       is_zero_1;
    
    // Wires for Term 2: e^(x - x_max)
    logic [7:0] e_a_2;
    logic [7:0] mantisa_2;
    logic       is_zero_2;
    
    // Evaluate e^(-x_max)
    tr_exp #(
        .FRAC_W(FRAC_W),
        .ITER(STAGE3_ITER)      // Zero-order approximation for TR-GELU
    ) u_exp_term1 (
        .x       (term1_reg), 
        .e_a     (e_a_1),               
        .mantisa (mantisa_1),           
        .is_zero (is_zero_1)
    );

    // Evaluate e^(x - x_max)
    tr_exp #(
        .FRAC(FRAC_W),
        .ITER(STAGE3_ITER)      // Zero-order approximation for TR-GELU
    ) u_exp_term2 (
        .x       (term2_reg), 
        .e_a     (e_a_2),               
        .mantisa (mantisa_2),           
        .is_zero (is_zero_2)
    );

    // --- Combinational Reconstruction ---
    logic [15:0] raw_exp1;
    logic [15:0] raw_exp2;
    logic [7:0]  final_exp1;
    logic [7:0]  final_exp2;

    always_comb begin
        raw_exp1 = e_a_1 * mantisa_1;
        raw_exp2 = e_a_2 * mantisa_2;

        if (STAGE3_ITER == 0) begin
            final_exp1 = e_a_1;
            final_exp2 = e_a_2;
        end else begin
            // Shift by 8 to get back to Q0.8
            final_exp1 = raw_exp1[FRAC_W + 7 : FRAC_W];
            final_exp2 = raw_exp2[FRAC_W + 7 : FRAC_W];
        end
    end
    
    // --- Sequential Accumulation ---
    logic        [8:0]   sum_S;     
    logic        [7:0]   exp_term2_reg; 
    logic signed [W-1:0] x_delay_s3;
    logic                valid_s3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_S         <= '0;
            exp_term2_reg <= '0;
            x_delay_s3    <= '0;
            valid_s3      <= 1'b0;
        end else begin
            valid_s3      <= valid_s2;
            sum_S         <= final_exp1 + final_exp2;
            exp_term2_reg <= final_exp2;    
            x_delay_s3    <= original_x_reg; 
        end
    end

    // ========================================================================
    // STAGE 4: Denominator Generator & Final Multiplier
    // ========================================================================
    logic [7:0] inv_S_comb;
    logic [7:0] inv_S;

    // Shift the 9-bit Q1.8 sum down by 4 bits to create a Q5.4 input
    logic [8:0] sum_S_q4;
    assign sum_S_q4 = sum_S >> 4;

    tr_reciprocal #(
        .IN_WIDTH  (9),      // sum_S is 9 bits (Q1.8)
        .OUT_WIDTH (8),      // Final output is 8 bits (Q0.8)
        .IN_FRAC   (4),      // Fractional precision of sum_S
        .OUT_FRAC  (4),      // Internal Log-domain working precision
        .INV_SQRT  (0),      // 0 = standard reciprocal (1/x)
        .ITER      (1)       // 1st-order Taylor for GELU division
    ) u_reciprocal (
        .clk   (clk),
        .rst_n (rst_n),
        .xq    (sum_S_q4),
        .yq    (inv_S_comb)
    );

    // --- Sequential Capture ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inv_S <= '0;
        end else begin
            // inv_S <= inv_S_comb;
            // Safety Clamp. If the mathematical sum is effectively 1.0 (256), 
            // force the reciprocal to 1.0 (255) to prevent 8-bit overflow truncation.
            inv_S <= (sum_S < 256) ? 8'hFF : inv_S_comb;
        end
    end

    // ------------------------------------------------------------------------
    // DELAY LINE: Synchronize the datapath
    // ------------------------------------------------------------------------
    // The tr_ln -> tr_exp -> reconstruction chain takes several clock cycles.
    // We must delay `x` and the numerator `e^(-x_max)` to arrive at the exact 
    // same time as `inv_S` for the final MAC.
    // NOTE: Adjust `RECIP_LATENCY` to match actual pipeline depth.

    localparam int RECIP_LATENCY = 1; 
    
    logic signed [W-1:0] x_shift           [RECIP_LATENCY];
    logic [7:0]          exp_term2_shift   [RECIP_LATENCY];
    logic                valid_shift       [RECIP_LATENCY];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < RECIP_LATENCY; i++) begin
                x_shift[i]         <= '0;
                exp_term2_shift[i] <= '0;
                valid_shift[i]     <= 1'b0;
            end
        end else begin
            x_shift[0]         <= x_delay_s3;
            exp_term2_shift[0] <= exp_term2_reg; 
            valid_shift[0]     <= valid_s3;
        end
    end

    // ------------------------------------------------------------------------
    // FINAL STAGE: GELU(x) = x * e^(x - x_max) * inv_S
    // ------------------------------------------------------------------------
    logic signed [25:0] gelu_mac; // 8-bit * 8-bit * 8-bit = 24-bit max growth

    always_comb begin
        gelu_mac = x_shift[RECIP_LATENCY-1] * $signed({1'b0, exp_term2_shift[RECIP_LATENCY-1]}) * $signed({1'b0, inv_S});
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gelu_out  <= '0;
            valid_out <= 1'b0;
        end else begin
            // Shift back to Q4.4 format
            gelu_out  <= gelu_mac[16 + W - 1 : 16]; 
            valid_out <= valid_shift[RECIP_LATENCY-1];
        end
    end

endmodule