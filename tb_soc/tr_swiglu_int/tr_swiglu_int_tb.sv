`timescale 1ns/1ps

// File-I/O testbench for the PRODUCTION SwiGLU opcode PAIR: drives
// tr_soc_top_int through CMD=0x05 (SiLU, SW_P1..P3) then CMD=0x06
// (gate-multiply, SW_P4) -- the real control-path integration added per
// docs/llama_fpga_plan.md's changes A (alpha_stabilizer flat_mode) and C
// (tr_nonlinear_vpu.sv's saturating vecmul_scale_mode=3'b100). Modeled
// directly on tr_gelu_int_tb.sv's load_vector()/mmio_write() pattern --
// CMD=5 needs "x" staged exactly like GELU's input (c_vec[0..3] via the
// requantizer passthrough, ext_sram_b[4..7] directly); CMD=6 needs "g"
// staged differently: ALL 8 lanes via ext_sram_b directly (SW_P4 reads
// sram_b with src_sram_b_sel=0, which routes every lane from ext_sram_b,
// unlike sram_a's req_vec_out/ext_sram_b M-lane split) -- no dot-product
// injection needed for g.
//
// Golden reference: swiglu_hw_scalar_bestfit() via cpu_math_model.cpp's
// "swiglu" mode (tr_math_model.hpp) -- bit-exact to what these two
// opcodes are DESIGNED to compute, confirmed independently against the
// compiled ctypes model and the Python model earlier in this work (0/20000
// mismatches). This is deliberately NOT tr_swiglu.sv's own precision (the
// "swiglu_baseline" mode/block) -- see docs/llama_fpga_plan.md for why.
module tb_tr_swiglu_int();

    localparam int M = 4;
    localparam int N = 8;
    localparam int W = 8;

    logic clk = 0, rst_n = 0;

    logic [7:0]  mmio_addr;
    logic [31:0] mmio_wdata;
    logic        mmio_wen;
    logic [31:0] mmio_rdata;

    logic        dot_in_valid, dot_in_ready;
    logic [W-1:0] a_mat [M][N];
    logic [W-1:0] b_vec [N];
    logic signed [31:0] c_vec [M];
    logic        clear_acc;

    logic signed [W-1:0] ext_sram_b [N];
    logic        vpu_out_valid;
    logic signed [W-1:0] vpu_data_out [N];

    logic [15:0] mm_tile_row, mm_tile_col;
    logic        mm_mem_rd;
    logic        mm_mem_valid;
    logic [W-1:0] mm_a_tile [M][N];
    logic [W-1:0] mm_b_tile [N];
    logic        mm_out_we;
    logic [15:0] mm_out_row;
    logic signed [W-1:0] mm_out_data [M];

    tr_soc_top_int #(.M(M), .N(N), .W(W), .ACC_W(32)) dut (.*);

    assign mm_mem_valid = 1'b1;
    always_comb begin
        for (int m = 0; m < M; m++)
            for (int i = 0; i < N; i++) mm_a_tile[m][i] = '0;
        for (int i = 0; i < N; i++) mm_b_tile[i] = '0;
    end

    always #5 clk = ~clk;

    task automatic mmio_write(input logic [7:0] addr, input logic [31:0] data);
        @(posedge clk);
        mmio_addr = addr; mmio_wdata = data; mmio_wen = 1;
        @(posedge clk); mmio_wen = 0;
    endtask

    task automatic wait_for_done();
        logic [31:0] status;
        do begin
            @(posedge clk);
            mmio_addr = 8'h04; // ADDR_STATUS
            mmio_wen  = 0;
            status = mmio_rdata;
        end while ((status & 32'h2) == 0);
    endtask

    // Stages "x": c_vec[0..3] + ext_sram_b[4..7], via the identity-requantizer
    // passthrough -- exactly tr_gelu_int_tb.sv's load_vector(), needed because
    // CMD=5's SW_P1 and SW_P3 both read x via req_vec_out/ext_sram_b
    // (src_sram_a_sel=0).
    task automatic load_x(input logic signed [W-1:0] x [N]);
        @(posedge clk);
        dot_in_valid = 1;
        clear_acc    = 1;
        for (int i = 0; i < M; i++) c_vec[i] = x[i];
        for (int i = 0; i < M; i++) ext_sram_b[i] = '0;
        for (int i = M; i < N; i++) ext_sram_b[i] = x[i];
        @(posedge clk);
        dot_in_valid = 0;
        do begin @(posedge clk); end while (!dut.req_out_valid);
        repeat (2) @(posedge clk);
    endtask

    // Stages "g": ALL N lanes directly via ext_sram_b -- CMD=6's SW_P4 reads
    // sram_b with src_sram_b_sel=0, which has no M-lane split (unlike
    // sram_a), so no dot-product injection is needed here at all.
    task automatic load_g(input logic signed [W-1:0] g [N]);
        for (int i = 0; i < N; i++) ext_sram_b[i] = g[i];
        @(posedge clk);
    endtask

    int file_in, file_out, num_vecs, dummy;
    int x_s[N], g_s[N];

    initial begin
        rst_n = 0; mmio_wen = 0; dot_in_valid = 0; clear_acc = 1;
        for (int i = 0; i < N; i++) begin b_vec[i] = '0; ext_sram_b[i] = '0; end
        for (int m = 0; m < M; m++) begin
            c_vec[m] = 0;
            for (int i = 0; i < N; i++) a_mat[m][i] = '0;
        end

        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_vecs);

        #22 rst_n = 1;

        // Requantizer pass-through: mult=1, shift=0 => c_vec passes unchanged
        // (needed for x's lanes 0..3 via the dot-product injection trick).
        mmio_write(8'h08, 32'd1);   // REQ_MULT
        mmio_write(8'h0C, 32'd0);   // REQ_SHIFT

        for (int v = 0; v < num_vecs; v++) begin
            logic signed [W-1:0] x [N];
            logic signed [W-1:0] g [N];
            dummy = $fscanf(file_in, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                x_s[0], x_s[1], x_s[2], x_s[3], x_s[4], x_s[5], x_s[6], x_s[7],
                g_s[0], g_s[1], g_s[2], g_s[3], g_s[4], g_s[5], g_s[6], g_s[7]);
            for (int i = 0; i < N; i++) x[i] = W'(x_s[i]);
            for (int i = 0; i < N; i++) g[i] = W'(g_s[i]);

            load_x(x);
            mmio_write(8'h00, 32'h05);  // CMD = SiLU (SwiGLU pass 1/2)
            wait_for_done();

            load_g(g);
            mmio_write(8'h00, 32'h06);  // CMD = gate-multiply (SwiGLU pass 2/2)
            wait_for_done();

            $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d\n",
                $signed(vpu_data_out[0]), $signed(vpu_data_out[1]),
                $signed(vpu_data_out[2]), $signed(vpu_data_out[3]),
                $signed(vpu_data_out[4]), $signed(vpu_data_out[5]),
                $signed(vpu_data_out[6]), $signed(vpu_data_out[7]));
        end

        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

endmodule
