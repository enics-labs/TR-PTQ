`timescale 1ns/1ps

// File-I/O testbench for the PRODUCTION fused "accumulator -> requantize ->
// GELU" sequence: drives tr_soc_top_int through CMD=0x02 (GELU), exactly as
// the firmware's tr_tensor.c does for the FFN gate
// (_xlr2_vpu(acc, CMD_GELU, ffn_gate_mult, ffn_gate_shift, tmp)):
//   c_vec[0..3]   = acc[0..3]      (raw INT32 matmul accumulator, M=4 lanes)
//   ext_sram_b[0..3] = 0            (zeroed, avoids contaminating vpu_sram_b)
//   ext_sram_b[4..7] = 0            (unused lanes -- firmware casts garbage
//                                     accumulator there, but only reads back
//                                     out[0..rm-1], rm<=M=4, so lanes 4..7
//                                     are don't-care for this fused-op check)
//   REQ_MULT/REQ_SHIFT = ffn_gate_mult/ffn_gate_shift (NOT a passthrough --
//                                     unlike tr_rmsnorm_int_tb.sv, this
//                                     exercises the real quantizer scale)
// a_mat=0, b_vec=0, clear_acc=1 => dot_product_engine's output = c_vec
// unchanged (bias passthrough), so req_vec_out = quant(acc, mult, shift) --
// this then feeds GL_P1..GL_P3 (see tr_soc_ctrl_int.sv) same as the
// standalone tr_gelu testbench, closing the loop on whether the FUSED
// firmware call composes exactly as quant() then gelu() in the unified
// Python/C++ model.
//
// Input format (matches cpu_math_model.cpp's "gelu_fused" mode):
//   acc0 acc1 acc2 acc3 mult shift   (INT32 acc/mult, INT shift)
// Output: out0 out1 out2 out3        (INT8, only the M=4 lanes that matter)
module tb_tr_gelu_fused_int();

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

    // Fires the accumulator load (c_vec[0..3]=acc, ext_sram_b=0) and waits
    // for the requantizer -- same handshake as tr_rmsnorm_int_tb.sv's
    // load_vector, but c_vec now carries the real INT32 accumulator instead
    // of an already-INT8 value.
    int req_dbg_file;

    task automatic load_acc(input logic signed [31:0] acc [M]);
        @(posedge clk);
        dot_in_valid = 1;
        clear_acc    = 1;
        for (int i = 0; i < M; i++) c_vec[i] = acc[i];
        for (int i = 0; i < N; i++) ext_sram_b[i] = '0;
        @(posedge clk);
        dot_in_valid = 0;
        do begin @(posedge clk); end while (!dut.req_out_valid);
        // Snapshot the requantizer's output HERE -- before it ever reaches
        // the GELU LUT sequence -- to bisect passthrough/requantize vs. the
        // GELU stage itself.
        $fwrite(req_dbg_file, "%0d %0d %0d %0d\n",
            $signed(dut.req_vec_out[0]), $signed(dut.req_vec_out[1]),
            $signed(dut.req_vec_out[2]), $signed(dut.req_vec_out[3]));
        repeat (2) @(posedge clk);
    endtask

    int file_in, file_out, num_vecs, dummy;
    longint acc_s[M]; int mult_s, shift_s;

    initial begin
        rst_n = 0; mmio_wen = 0; dot_in_valid = 0; clear_acc = 1;
        for (int i = 0; i < N; i++) begin b_vec[i] = '0; ext_sram_b[i] = '0; end
        for (int m = 0; m < M; m++) begin
            c_vec[m] = 0;
            for (int i = 0; i < N; i++) a_mat[m][i] = '0;
        end

        file_in     = $fopen("inputs.txt",  "r");
        file_out    = $fopen("hdl_out.txt", "w");
        req_dbg_file = $fopen("req_dbg.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_vecs);

        #22 rst_n = 1;

        for (int v = 0; v < num_vecs; v++) begin
            logic signed [31:0] acc [M];
            dummy = $fscanf(file_in, "%d %d %d %d %d %d\n",
                acc_s[0], acc_s[1], acc_s[2], acc_s[3], mult_s, shift_s);
            for (int i = 0; i < M; i++) acc[i] = 32'(acc_s[i]);

            mmio_write(8'h08, 32'(mult_s));   // REQ_MULT
            mmio_write(8'h0C, 32'(shift_s));  // REQ_SHIFT

            load_acc(acc);

            mmio_write(8'h00, 32'h02);  // CMD = GELU
            wait_for_done();

            $fwrite(file_out, "%0d %0d %0d %0d\n",
                $signed(vpu_data_out[0]), $signed(vpu_data_out[1]),
                $signed(vpu_data_out[2]), $signed(vpu_data_out[3]));
        end

        $fclose(file_in);
        $fclose(file_out);
        $fclose(req_dbg_file);
        $finish;
    end

endmodule
