`timescale 1ns/1ps

module tr_gelu #(
    parameter int W = 8  // Q4.4 format (1 sign, 3 int, 4 frac)
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 valid_in,
    input  logic signed [W-1:0]  x_in,

    output logic                 valid_out,
    output logic signed [W-1:0]  gelu_out
);
    localparam int FRAC_W = 4;

    // --- Internal Signal Declarations ---
    logic signed [W-1:0] alpha_g, x_scaled_comb, z_s1, x_s1;
    logic signed [W-1:0] u_neg_abs, x_s2, z_s2;
    logic signed [W-1:0] x_s3, z_s3;
    logic [7:0]          e_a, e_man, E_s3;
    logic                is_zero, valid_s1, valid_s2, valid_s3;

    // ========================================================================
    // STAGE 1: Scaling (x = alpha * z)
    // ========================================================================
    always_comb begin
        logic [7:0] abs_z;
        logic signed [15:0] x_mult;
        abs_z = (x_in[W-1]) ? -x_in : x_in;
        
        // Region-dependent scaling coefficients
        case (abs_z[6:4]) 
            3'b000:  alpha_g = 8'h1B; // ~1.702
            3'b001:  alpha_g = 8'h1A;
            3'b010:  alpha_g = 8'h19;
            default: alpha_g = 8'h18;
        endcase
        
        x_mult = x_in * $signed({1'b0, alpha_g});
        
        // Saturation to prevent wrapping
        if ($signed(x_mult) > 16'sd2032)       x_scaled_comb = 8'sd127;
        else if ($signed(x_mult) < -16'sd2048) x_scaled_comb = -8'sd128;
        else                                   x_scaled_comb = x_mult[11:4]; 
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_s1 <= '0; z_s1 <= '0; valid_s1 <= 1'b0;
        end else begin
            x_s1 <= x_scaled_comb;
            z_s1 <= x_in;
            valid_s1 <= valid_in;
        end
    end

    // ========================================================================
    // STAGE 2: Numerical Stabilization (u = -|x|)
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            u_neg_abs <= '0; x_s2 <= '0; z_s2 <= '0; valid_s2 <= 1'b0;
        end else begin
            // Reusing one ALU: Compute e^-|x| for all cases
            u_neg_abs <= (x_s1 > 0) ? -x_s1 : x_s1;
            x_s2      <= x_s1;
            z_s2      <= z_s1;
            valid_s2  <= valid_s1;
        end
    end

    // ========================================================================
    // STAGE 3: Single Exponential (E = e^-|x|)
    // ========================================================================
    tr_exp #(.FRAC(FRAC_W), .ITER(0)) u_exp (
        .x(u_neg_abs), .e_a(e_a), .mantisa(e_man), .is_zero(is_zero)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            E_s3 <= '0; x_s3 <= '0; z_s3 <= '0; valid_s3 <= 1'b0;
        end else begin
            E_s3 <= is_zero ? 8'd0 : e_a; 
            x_s3 <= x_s2;
            z_s3 <= z_s2;
            valid_s3 <= valid_s2;
        end
    end

    // ========================================================================
    // STAGE 4: Reciprocal Denominator
    // ========================================================================
    logic [7:0] inv_S_q4; 
    wire [8:0] sum_S = 9'd256 + E_s3; 
    
    tr_reciprocal #(.IN_WIDTH(9), .OUT_WIDTH(8), .IN_FRAC(4), .OUT_FRAC(4), .ITER(1)) 
    u_recip (.clk(clk), .rst_n(rst_n), .xq(sum_S >> 4), .yq(inv_S_q4));

    // Pipeline Sync: Match the sequential delay of tr_reciprocal (~4 cycles)
    localparam int SYNC = 4;
    logic signed [W-1:0] z_pipe [SYNC], x_pipe [SYNC];
    logic [7:0]          E_pipe [SYNC];
    logic                v_pipe [SYNC];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for(int i=0; i<SYNC; i++) begin 
                z_pipe[i] <= '0; x_pipe[i] <= '0; E_pipe[i] <= '0; v_pipe[i] <= 0; 
            end
        end else begin
            z_pipe[0] <= z_s3; x_pipe[0] <= x_s3; E_pipe[0] <= E_s3; v_pipe[0] <= valid_s3;
            for(int i=1; i<SYNC; i++) begin
                z_pipe[i] <= z_pipe[i-1]; x_pipe[i] <= x_pipe[i-1]; 
                E_pipe[i] <= E_pipe[i-1]; v_pipe[i] <= v_pipe[i-1];
            end
        end
    end

    // Final Stage Signal Taps
    wire signed [W-1:0] z_s4 = z_pipe[SYNC-1];
    wire signed [W-1:0] x_s4 = x_pipe[SYNC-1];
    wire [7:0]          E_s4 = E_pipe[SYNC-1];
    wire                v_s4 = v_pipe[SYNC-1];

    // ========================================================================
    // STAGE 5: Selection & Final Product
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            gelu_out <= '0; valid_out <= 1'b0;
        end else begin
            logic [7:0]  sigmoid_q8;
            logic [15:0] sig_neg_mult;
            logic signed [15:0] gelu_prod;

            // inv_S_q4 is already the Q0.8 reciprocal result
            if (x_s4 > 0) begin
                sigmoid_q8 = inv_S_q4; 
            end else begin
                // sig = E * (1/S) -> Q0.8 * Q0.8 = Q0.16 -> Shift to Q0.8
                sig_neg_mult = (E_s4 * inv_S_q4);
                sigmoid_q8   = sig_neg_mult[15:8];
            end

            gelu_prod = z_s4 * $signed({1'b0, sigmoid_q8}); // Q4.4 * Q0.8 = Q4.12
            // Use truncation to match the 'Original' report's accuracy
            gelu_out  <= gelu_prod[15:8]; 
            valid_out <= v_s4;
        end
    end
endmodule