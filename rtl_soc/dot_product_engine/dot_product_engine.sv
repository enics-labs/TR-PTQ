/*
 * @module   dot_product_engine
 * @brief    Dense Matrix-Vector Compute Wrapper.
 * @details  Instantiates M parallel `mac_array_engine` lanes. 
 *           Computes: Out[i] = SUM(A[i] * B) + C[i].
 *           Uses a broadcast architecture where Vector B is shared across all lanes.
 */
module dot_product_engine #(
    parameter int M     = 4,  // Number of parallel output lanes (Matrix Height)
    parameter int N     = 16, // Dot product vector dimension (Matrix Width)
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic clk,
    input  logic rst_n,

    // Input stream
    input  logic in_valid,
    output logic in_ready,
    input  logic [1:0]               op_mode,
    input  logic [W-1:0]             a_mat [M][N], // M independent vectors (Weights)
    input  logic [W-1:0]             b_vec [N],    // 1 shared vector (Activations)
    input  logic signed [ACC_W-1:0]  c_vec [M],    // Independent lane biases
    input  logic clear_acc,

    // Output stream
    output logic out_valid,
    input  logic out_ready,
    output logic signed [ACC_W-1:0]  out_vec [M]   // M parallel dot product results
);

    // Because all M lanes share the exact same valid/ready pipeline timing,
    // we only need to monitor the handshake signals of Lane 0.
    logic [M-1:0] lane_in_ready;
    logic [M-1:0] lane_out_valid;

    assign in_ready  = lane_in_ready[0];
    assign out_valid = lane_out_valid[0];

    // =========================================================
    // PARALLEL MAC GENERATION
    // =========================================================
    generate
        for (genvar i = 0; i < M; i++) begin : GEN_MAC_LANES
            mac_array_engine #(
                .N(N), .W(W), .ACC_W(ACC_W)
            ) u_mac_lane (
                .clk       (clk),
                .rst_n     (rst_n),
                .in_valid  (in_valid),
                .in_ready  (lane_in_ready[i]),
                .op_mode   (op_mode),
                .a         (a_mat[i]),       // Lane i receives row i of Matrix A
                .b         (b_vec),          // Vector B is broadcast to all lanes
                .c         (c_vec[i]),       // Lane i receives bias i
                .clear_acc (clear_acc),
                .out_valid (lane_out_valid[i]),
                .out_ready (out_ready),
                .out_dot   (out_vec[i])      // Lane i outputs result i
            );
        end
    endgenerate

endmodule