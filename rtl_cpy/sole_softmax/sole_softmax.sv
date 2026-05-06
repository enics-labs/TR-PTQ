`timescale 1ns/1ps

module sole_softmax #(
    parameter int N = 8,
    parameter int W = 8,
    parameter int FRAC_W = 4
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Input Stream
    input  logic                    in_valid,
    input  logic signed [W-1:0]     in_data [N],
    
    // Normalized Output Stream
    output logic                    out_valid,
    output logic        [W-1:0]     out_data [N] 
);

    localparam int MAX_LATENCY = $clog2(N); // 3 cycles for N=8
    localparam int SUM_W = W + $clog2(N);   // 8 + 3 = 11 bits for Sum

    // ========================================================================
    // STAGE 1: Pipelined Max Finding & Input Delay Line
    // ========================================================================
    logic signed [W-1:0] x_max;
    logic                max_valid;
    logic signed [W-1:0] a_delayed [MAX_LATENCY+1][N];

    piped_max #(
        .NUM_INPUTS(N), .DATA_WIDTH(W)
    ) u_max_tree (
        .clk(clk), .rst_n(rst_n), 
        .valid_in(in_valid), .in_data(in_data),
        .max_out(x_max), .valid_out(max_valid)
    );

    // Delay the inputs to match the Max Tree arrival
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s <= MAX_LATENCY; s++) begin
                for (int i = 0; i < N; i++) a_delayed[s][i] <= '0;
            end
        end else begin
            a_delayed[0] <= in_data;
            for (int s = 1; s <= MAX_LATENCY; s++) begin
                a_delayed[s] <= a_delayed[s-1];
            end
        end
    end

    // ========================================================================
    // STAGE 2: Max Subtraction, SOLE Exponentiation, and Summation
    // ========================================================================
    logic signed [W-1:0] x_sub [N];
    logic        [W-1:0] exp_comb [N];
    logic        [SUM_W-1:0] sum_comb;

    // A. Max Subtraction (Combinational)
    max_sub #(
        .NUM_INPUTS(N), .DATA_WIDTH(W)
    ) u_sub (
        .in_data(a_delayed[MAX_LATENCY-1]), .x_max(x_max), .out_data(x_sub)
    );

    // B. SOLE Exponentiation Lanes (Combinational)
    generate
        for (genvar i = 0; i < N; i++) begin : GEN_EXP
            sole_log2exp #(.W(W), .FRAC_W(FRAC_W)) u_exp (
                .x(x_sub[i]), .exp_out(exp_comb[i])
            );
        end
    endgenerate

    // C. Denominator Summation Tree (Combinational)
    always_comb begin
        sum_comb = '0;
        for (int i = 0; i < N; i++) begin
            sum_comb = sum_comb + exp_comb[i];
        end
    end

    // D. Stage 2 Registers
    logic             stg2_valid;
    logic [W-1:0]     exp_reg [N];
    logic [SUM_W-1:0] sum_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg2_valid <= 1'b0;
            sum_reg    <= '0;
            for (int i = 0; i < N; i++) exp_reg[i] <= '0;
        end else begin
            stg2_valid <= max_valid; // Flows from piped_max
            sum_reg    <= sum_comb;
            exp_reg    <= exp_comb;
        end
    end

    // ========================================================================
    // STAGE 3: Approximate Log Division (ALDIV)
    // ========================================================================
    logic [4:0]  lod_k;
    logic [3:0]  lod_s_frac;
    logic [15:0] recip_approx;

    always_comb begin
        lod_k = '0;
        
        // 1. Find Leading One of the Sum
        for (int i = 0; i < SUM_W; i++) begin
            if (sum_reg[i]) lod_k = i[4:0];
        end
        
        // 2. Extract 4-bit fraction cleanly using a padded shift
        // Appending 4 zeros to the bottom and shifting right by lod_k 
        // perfectly aligns the 4 fraction bits into the LSBs, avoiding illegal variable part-selects.
        lod_s_frac = ({sum_reg, 4'b0000} >> lod_k);

        // 3. Approximate 1 / (1+s) -> 1.0 proxy is 16'h1000
        recip_approx = 16'h1000 - (lod_s_frac << 7); 
    end

    // Stage 3 Registers
    logic        stg3_valid;
    logic [W-1:0] exp_reg2 [N];
    logic [4:0]  lod_k_reg;
    logic [15:0] recip_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stg3_valid <= 1'b0;
            lod_k_reg  <= '0;
            recip_reg  <= '0;
            for (int i = 0; i < N; i++) exp_reg2[i] <= '0;
        end else begin
            stg3_valid <= stg2_valid;
            lod_k_reg  <= lod_k;
            recip_reg  <= recip_approx;
            exp_reg2   <= exp_reg;
        end
    end

    // ========================================================================
    // STAGE 4: Final Normalization (Multiply & Shift)
    // ========================================================================
    // No full MAC required, just a final parallel normalization pass.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            for (int i = 0; i < N; i++) out_data[i] <= '0;
        end else begin
            out_valid <= stg3_valid;
            
            for (int i = 0; i < N; i++) begin
                // Q4.4 Exponent * Q4.12 Reciprocal Proxy = Q8.16 Product
                logic [23:0] prod;
                prod = exp_reg2[i] * recip_reg;
                
                // Shift down by fractional width (12 + 4) and adjust for LOD Scale
                out_data[i] = prod >> (lod_k_reg + 8); 
            end
        end
    end

endmodule