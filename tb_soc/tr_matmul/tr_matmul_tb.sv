`timescale 1ns/1ps

// File-I/O testbench chaining dot_product_engine (M=4, N=8) directly into
// requantize_engine_int (N=4) -- the production tr_soc_top_int single-tile
// path (mac_array + requantizer).  Matches verify_block.py's convention
// (inputs.txt / hdl_out.txt) and cpu_math_model.cpp's "matmul" mode:
//   y[m] = quant(c[m] + sum_k(A[m][k]*b[k]), mult, shift)
module tb_tr_matmul();

    localparam int M       = 4;
    localparam int N       = 8;
    localparam int W       = 8;
    localparam int ACC_W   = 32;
    localparam int MUL_W   = 32;
    localparam int SHIFT_W = 6;
    localparam int OUT_W   = 8;

    logic clk = 0, rst_n = 0;

    // dot_product_engine
    logic dot_in_valid, dot_in_ready;
    logic [1:0] op_mode;
    logic [W-1:0] a_mat [M][N];
    logic [W-1:0] b_vec [N];
    logic signed [ACC_W-1:0] c_vec [M];
    logic clear_acc;
    logic dot_out_valid, dot_out_ready;
    logic signed [ACC_W-1:0] dot_out_vec [M];

    dot_product_engine #(.M(M), .N(N), .W(W), .ACC_W(ACC_W)) u_dot (
        .clk(clk), .rst_n(rst_n),
        .in_valid(dot_in_valid), .in_ready(dot_in_ready),
        .op_mode(op_mode), .a_mat(a_mat), .b_vec(b_vec), .c_vec(c_vec),
        .clear_acc(clear_acc),
        .out_valid(dot_out_valid), .out_ready(dot_out_ready), .out_vec(dot_out_vec)
    );

    // requantize_engine_int, fed directly from the dot product's output
    logic signed [MUL_W-1:0] multiplier;
    logic [SHIFT_W-1:0]      shift;
    logic req_out_valid, req_out_ready;
    logic signed [OUT_W-1:0] req_out_vec [M];

    assign dot_out_ready = 1'b1;   // requantizer is always ready (see quant tb)

    requantize_engine_int #(
        .N(M), .ACC_W(ACC_W), .MUL_W(MUL_W), .SHIFT_W(SHIFT_W), .OUT_W(OUT_W)
    ) u_req (
        .clk(clk), .rst_n(rst_n),
        .in_valid(dot_out_valid), .in_ready(),
        .acc_in(dot_out_vec), .multiplier(multiplier), .shift(shift),
        .out_valid(req_out_valid), .out_ready(req_out_ready), .out_vec(req_out_vec)
    );
    assign req_out_ready = 1'b1;

    always #5 clk = ~clk;

    int file_in, file_out, num_vecs, dummy;
    int a_s[M*N], b_s[N], c_s[M], mult_s, shift_s;

    initial begin
        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_vecs);

        dot_in_valid = 0; op_mode = 2'd0; clear_acc = 1;
        for (int m = 0; m < M; m++) begin
            c_vec[m] = '0;
            for (int k = 0; k < N; k++) a_mat[m][k] = '0;
        end
        for (int k = 0; k < N; k++) b_vec[k] = '0;
        multiplier = '0; shift = '0;
        #20; rst_n = 1;
        @(posedge clk);

        for (int v = 0; v < num_vecs; v++) begin
            for (int i = 0; i < M*N; i++) dummy = $fscanf(file_in, "%d", a_s[i]);
            for (int i = 0; i < N;   i++) dummy = $fscanf(file_in, "%d", b_s[i]);
            for (int i = 0; i < M;   i++) dummy = $fscanf(file_in, "%d", c_s[i]);
            dummy = $fscanf(file_in, "%d %d\n", mult_s, shift_s);

            for (int m = 0; m < M; m++) begin
                for (int k = 0; k < N; k++) a_mat[m][k] = W'(a_s[m*N + k]);
                c_vec[m] = ACC_W'(c_s[m]);
            end
            for (int k = 0; k < N; k++) b_vec[k] = W'(b_s[k]);
            multiplier = MUL_W'(mult_s);
            shift      = SHIFT_W'(shift_s);

            @(negedge clk); dot_in_valid = 1;
            @(negedge clk); dot_in_valid = 0;
            while (!req_out_valid) @(posedge clk);

            $fwrite(file_out, "%0d %0d %0d %0d\n",
                $signed(req_out_vec[0]), $signed(req_out_vec[1]),
                $signed(req_out_vec[2]), $signed(req_out_vec[3]));

            @(posedge clk);
        end

        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

endmodule