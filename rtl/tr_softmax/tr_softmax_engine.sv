module tr_softmax_engine #(
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
    // 1. Core TR-Softmax Engine (Decomposition & Reciprocal)
    // ========================================================================
    logic                 valid_decomp;
    logic [W-1:0]         e_a [N];
    logic [W-1:0]         e_frac [N];
    logic                 valid_sum;
    logic [7:0]           inv_S;

    tr_softmax #(
        .N(N), .W(W), .ACC_W(ACC_W)
    ) u_tr_softmax (
        .clk              (clk),
        .rst_n            (rst_n),
        .valid_in         (valid_in),
        .in_data          (in_data),
        .out_valid_decomp (valid_decomp),
        .out_e_a          (e_a),
        .out_e_frac       (e_frac),
        .out_valid_sum    (valid_sum),
        .out_inv_S        (inv_S)
    );

    // ========================================================================
    // 2. The Synchronization Delay Line
    // ========================================================================
    // The vector dot-product MAC takes 4 clock cycles to compute the sum.
    // We must delay the raw exponentials so they arrive at the exact same 
    // time as the final reciprocal.
    localparam int MAC_LATENCY = 4; 
    
    logic [W-1:0] e_a_delayed    [MAC_LATENCY][N];
    logic [W-1:0] e_frac_delayed [MAC_LATENCY][N];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s < MAC_LATENCY; s++) begin
                for (int i = 0; i < N; i++) begin
                    e_a_delayed[s][i] <= '0;
                    e_frac_delayed[s][i] <= '0;
                end
            end
        end else begin
            e_a_delayed[0]    <= e_a;
            e_frac_delayed[0] <= e_frac;
            
            for (int s = 1; s < MAC_LATENCY; s++) begin
                e_a_delayed[s]    <= e_a_delayed[s-1];
                e_frac_delayed[s] <= e_frac_delayed[s-1];
            end
        end
    end

    // Create a clean wire alias for the final delayed outputs
    wire [W-1:0] final_e_a    [N] = e_a_delayed[MAC_LATENCY-1];
    wire [W-1:0] final_e_frac [N] = e_frac_delayed[MAC_LATENCY-1];

    // ========================================================================
    // 3. Final Requantization & Scaling (Strict IEEE SystemVerilog)
    // ========================================================================
    logic [23:0] e_x [N];
    logic [31:0] scaled_prob [N];

    // Strict Compliant Combinational Logic (Isolated from Flip-Flops)
    always_comb begin
        for (int i = 0; i < N; i++) begin
            // 1. Recombine e^x: (e_a * e_frac)
            e_x[i] = {16'd0, final_e_a[i]} * {16'd0, final_e_frac[i]};
            
            // 2. Multiply by reciprocal: e^x * (1/S)
            scaled_prob[i] = e_x[i] * {24'd0, inv_S};
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            for (int i = 0; i < N; i++) prob_out[i] <= '0;
        end else begin
            valid_out <= valid_sum; 
            
            for (int i = 0; i < N; i++) begin
                if (valid_sum) begin
                    // 3. Shift >> 12 to normalize back to an 8-bit Qx.8 integer probability.
                    // If the probability mathematically exceeds 1.0 (bit 20 high), clamp to 0xFF.
                    prob_out[i] <= scaled_prob[i][20] ? 8'hFF : scaled_prob[i][19:12];
                end
            end
        end
    end

endmodule