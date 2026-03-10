`timescale 1ns/1ps

module softmax_engine #(
    parameter int N = 8,
    parameter int W = 8,
    parameter int ACC_W = 32
)(
    input  logic                 clk,
    input  logic                 rst_n,
    
    // Input Stream (Raw Attention Scores)
    input  logic                 valid_in,
    input  logic signed [W-1:0]  in_data [N],
    
    // Output Stream (SoftMax Probabilities)
    output logic                 valid_out,
    output logic signed [W-1:0]  prob_out [N]
);

    // ========================================================================
    // STAGE 1: Max Scale & TR-Decomposition
    // ========================================================================
    // Subtract the max value to prevent overflow, then decompose the 
    // numbers into the Taylor-Region anchor (e_a) and mantissa (e_frac).
    
    logic                   valid_decomp;
    logic signed [W-1:0]    e_a    [N];
    logic signed [W-1:0]    e_frac [N];

    exp_x_minus_xmax #(
        .NUM_INPUTS(N),
        .DATA_WIDTH(W)
    ) u_decompose (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .in_data   (in_data),
        .valid_out (valid_decomp),
        .e_a       (e_a),
        .e_frac    (e_frac)
    );

    // ========================================================================
    // STAGE 2: Exponential Summation (The Denominator S)
    // ========================================================================
    // Reuse Vector MAC engine to multiply e_a * e_frac and accumulate 
    // the results across the vector.

    logic                    valid_sum;
    logic signed [ACC_W-1:0] sum_S;

    // Turn e_a and e_frac from signed to unsigned
    logic [W-1:0] mac_in1 [N];
    logic [W-1:0] mac_in2 [N];

    always_comb begin
        for (int i = 0; i < N; i++) begin
            mac_in1[i] = e_a[i];
            mac_in2[i] = e_frac[i];
        end
    end

    vec_mac_dsp48 #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) dsp_mac (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(valid_decomp),
        .in_ready(),
        .a(mac_in1),
        .b(mac_in2),
        .clear_acc(valid_decomp), // Clear accumulator on every new vector
        .op_mode(2'b10),        // Mode 2'b10: Unsigned x Unsigned
        .out_valid(valid_sum),
        .out_ready(1'b1),
        .out_dot(sum_S)
    );

    // // ========================================================================
    // // STAGE 2: Exponential Summation (The Denominator S)
    // // ========================================================================
    // // Reuse Vector MAC engine to multiply e_a * e_frac and accumulate 
    // // the results across the vector. Latency = 3 clock cycles.
    
    // logic                    valid_sum;
    // logic signed [ACC_W-1:0] sum_S;

    // exp_sum #(
    //     .N(N),
    //     .W(W),
    //     .ACC_W(ACC_W)
    // ) u_exp_sum (
    //     .clk       (clk),
    //     .rst_n     (rst_n),
    //     .in_valid  (valid_decomp),
    //     .in_ready  (),               // Ignoring backpressure for a pure pipeline
    //     .clear_acc (valid_decomp),   // Clear accumulator on every new vector
    //     .op_type   (1'b1),           // Mode 1: TR-EXP MAC Mode
    //     .x1        (e_a),
    //     .x2        (e_frac),
    //     .out_valid (valid_sum),
    //     .out_ready (1'b1),
    //     .out_dot   (sum_S)
    // );

    // ========================================================================
    // STAGE 3: The Synchronization Delay Line
    // ========================================================================
    // While `exp_sum` spends 6 clock cycles calculating the total denominator S,
    // we must buffer the individual e_a and e_frac values so they arrive at 
    // the final multiplier at the exact same moment the reciprocal finishes.
    
    localparam int MAC_LATENCY = 4; 
    logic signed [W-1:0] e_a_delayed    [MAC_LATENCY][N];
    logic signed [W-1:0] e_frac_delayed [MAC_LATENCY][N];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < MAC_LATENCY; s++) begin
                for (int i = 0; i < N; i++) begin
                    e_a_delayed[s][i] <= '0;
                    e_frac_delayed[s][i] <= '0;
                end
            end
        end else begin
            // FIX: Unconditional shift. The data must flow continuously 
            // to stay perfectly parallel with the MAC engine.
            e_a_delayed[0]    <= e_a;
            e_frac_delayed[0] <= e_frac;
            
            for (int s = 1; s < MAC_LATENCY; s++) begin
                e_a_delayed[s]    <= e_a_delayed[s-1];
                e_frac_delayed[s] <= e_frac_delayed[s-1];
            end
        end
    end

    // Create a clean wire alias for the final delayed outputs
    wire signed [W-1:0] final_e_a    [N] = e_a_delayed[MAC_LATENCY-1];
    wire signed [W-1:0] final_e_frac [N] = e_frac_delayed[MAC_LATENCY-1];

    // ========================================================================
    // STAGE 4: Denominator Generator (Reciprocal 1/S)
    // ========================================================================
    // Converts the 32-bit integer sum into the normalization scaling factor.
    // This is purely combinational logic, resolving instantly when sum_S arrives.
    
    logic [7:0] inv_S_raw;
    logic [7:0] inv_S;

    tr_reciprocal #(
        .WIDTH(16),
        .OUT_WIDTH(8)
    ) u_reciprocal (
        .clk   (clk),
        .rst_n (rst_n),
        .xq    (sum_S[23:8]),
        .yq    (inv_S_raw)
    );

    assign inv_S = (sum_S[23:8] <= 16) ? 8'hFF : inv_S_raw;

    // ========================================================================
    // STAGE 5: Final Requantization & Scaling (Prob = e^x * 1/S)
    // ========================================================================
    // Reassemble the individual exponentials and multiply by the reciprocal.
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (int i = 0; i < N; i++) prob_out[i] <= '0;
        end else begin
            valid_out <= valid_sum; // Probabilities are valid 1 cycle after the sum
            
            for (int i = 0; i < N; i++) begin
                if (valid_sum) begin
                    // 1. Recombine e^x: (e_a * e_frac)
                    // 2. Multiply by reciprocal: * inv_S
                    // 3. Shift >> 12 to normalize back to an 8-bit integer probability
                    
                    automatic logic [23:0] e_x = {16'd0, final_e_a[i]} * {16'd0, final_e_frac[i]};
                    automatic logic [31:0] scaled_prob = e_x * {24'd0, inv_S};
                    
                    // Simple truncation scaling (Replaces the complex requant_unit for SoftMax)
                    // prob_out[i] <= scaled_prob[19:12]; 
                    prob_out[i] <= scaled_prob[20] ? 8'hFF : scaled_prob[19:12];
                end
            end
        end
    end

endmodule