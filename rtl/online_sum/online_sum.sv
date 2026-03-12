// assuming power of two vector size

module exp_x_minus_xmax #(
    parameter int NUM_INPUTS = 8,
    parameter int DATA_WIDTH = 8,
    parameter int ITER = 1
    // parameter int MAX_LATENCY = 3   // latency of pipelined_max_tree
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          valid_in,
    input  logic signed [DATA_WIDTH-1:0]  in_data [NUM_INPUTS],

    output logic                          valid_out,
    output logic signed [DATA_WIDTH-1:0]  e_a [NUM_INPUTS],  // widened
    output logic signed [DATA_WIDTH-1:0]  e_frac [NUM_INPUTS]  // widened
);

localparam int MAX_LATENCY = $clog2(NUM_INPUTS) - 1;

// -----------------------------------------------
// Stage 1: Max tree
// -----------------------------------------------
logic signed [DATA_WIDTH-1:0] x_max;
logic                         max_valid;

pipelined_max_tree #(
    .NUM_INPUTS (NUM_INPUTS),
    .DATA_WIDTH (DATA_WIDTH)
) u_max_tree (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in (valid_in),
    .in_data  (in_data),
    .max_out  (x_max),
    .valid_out(max_valid)
);

// -----------------------------------------------
// Stage 2: Delay input vector to match x_max
// -----------------------------------------------
logic signed [DATA_WIDTH-1:0] in_data_d [MAX_LATENCY+1][NUM_INPUTS];
logic                         valid_d   [MAX_LATENCY+1];

integer i, i2, i3, j, k, s;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (s = 0; s < MAX_LATENCY; s++) begin
            valid_d[s] <= 1'b0;
        end
        for (i3 = 0; i3 < MAX_LATENCY+1; i3++)
            for (i2 = 0; i2 < NUM_INPUTS; i2++)
                in_data_d[i3][i2] <= 8'd0;
    end else begin
        valid_d[0] <= valid_in;
        for (i = 0; i < NUM_INPUTS; i++)
            in_data_d[0][i] <= in_data[i];

        for (s = 1; s < MAX_LATENCY+1; s++) begin
            valid_d[s] <= valid_d[s-1];
            for (i = 0; i < NUM_INPUTS; i++)
                in_data_d[s][i] <= in_data_d[s-1][i];
        end
    end
end

wire signed [DATA_WIDTH-1:0] aligned_data [NUM_INPUTS];
assign aligned_data = in_data_d[MAX_LATENCY];

// -----------------------------------------------
// Stage 3: Compute x - x_max
// -----------------------------------------------
logic signed [DATA_WIDTH:0] x_shifted [NUM_INPUTS];
logic signed [DATA_WIDTH-1:0] x_shifted_clamped [NUM_INPUTS];

// -----------------------------------------------
// Stage 3: Compute x - x_max with proper bit growth
// -----------------------------------------------
// logic signed [DATA_WIDTH:0] x_shifted [NUM_INPUTS]; // 9 bits

always_comb begin
    for (int j = 0; j < NUM_INPUTS; j++) begin
        // Sign-extend to 9 bits BEFORE subtracting to prevent 8-bit wrap
        // x_shifted[j] = $signed(aligned_data[j]) - $signed(x_max);
        x_shifted[j] = $signed({aligned_data[j][DATA_WIDTH-1], aligned_data[j]}) - 
                       $signed({x_max[DATA_WIDTH-1], x_max});
        
        // Clamp logic: since x <= x_max, x_shifted is always <= 0.
        // We only need to clamp the underflow (most negative value).
        if (x_shifted[j] < -128) begin
            x_shifted_clamped[j] = -128;
        end else begin
            x_shifted_clamped[j] = x_shifted[j][DATA_WIDTH-1:0];
        end
    end
end
// -----------------------------------------------
// Stage 4: TR-EXP per element (fully parallel)
// -----------------------------------------------
genvar g;
generate
    for (g = 0; g < NUM_INPUTS; g++) begin : GEN_EXP
        tr_exp #(
            .ITER (ITER)
        ) u_tr_exp (
            .x       (x_shifted_clamped[g]), // signed Q4, ≤ 0
            .e_a     (e_a[g]),               // Q1.7 or Q8
            .mantisa (e_frac[g]),            // Q1.7 or Q8
            .is_zero ()
        );
    end
endgenerate

// -----------------------------------------------
// Stage 5: Output formatting + valid
// -----------------------------------------------
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out <= 1'b0;
    end else begin
        valid_out <= valid_d[MAX_LATENCY-1]; // adjust if tr_exp pipelined
    end
end

endmodule
