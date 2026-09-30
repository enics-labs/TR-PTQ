`timescale 1ns/1ps

// File-I/O testbench for the PRODUCTION SOFTMAX sequence: drives
// tr_soc_top_int through CMD=0x01 (SOFTMAX), the SM_P1..SM_P4 FSM in
// tr_soc_ctrl_int.sv sharing the tr_nonlinear_vpu backbone. This is the
// module firmware actually exercises -- NOT tr_softmax.sv
// (tb_baseline/tr_baseline/tr_softmax/), which is a separate,
// differently-architected module (explicit reciprocal-via-exp instead of
// this FSM's log-sum-exp trick) not instantiated anywhere in
// tr_soc_top_int.sv. Same trap tr_gelu.sv/tr_rmsnorm.sv turned out to be
// -- see tr_gelu_int_tb.sv.
//
// Output is Q0.8 UNSIGNED (0..255, saturating) -- matches SM_P4's
// vecmul_scale_mode==2'b11 path and cpu_math_model.cpp's "softmax" mode
// (softmax_hw_model). Follows the same requantizer-passthrough convention
// as tr_gelu_int_tb.sv / tr_rmsnorm_int_tb.sv: mult=1, shift=0 so c_vec
// passes through unchanged, x[0..3] via c_vec (lanes 0..M-1), x[4..7] via
// ext_sram_b (lanes M..N-1).
module tb_tr_softmax_int();

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

    // Fires the passthrough MAC (c_vec[0..3] + ext_sram_b[4..7]) and waits for
    // the requantizer, exactly as tr_gelu_int_tb.sv's load_vector does.
    task automatic load_vector(input logic signed [W-1:0] x [N]);
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

    int file_in, file_out, stage_dbg_file, num_vecs, dummy;
    int x_s[N];

    initial begin
        rst_n = 0; mmio_wen = 0; dot_in_valid = 0; clear_acc = 1;
        for (int i = 0; i < N; i++) begin b_vec[i] = '0; ext_sram_b[i] = '0; end
        for (int m = 0; m < M; m++) begin
            c_vec[m] = 0;
            for (int i = 0; i < N; i++) a_mat[m][i] = '0;
        end

        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        stage_dbg_file = $fopen("stage_dbg.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_vecs);

        #22 rst_n = 1;

        // Requantizer pass-through: mult=1, shift=0 => c_vec passes unchanged.
        mmio_write(8'h08, 32'd1);   // REQ_MULT
        mmio_write(8'h0C, 32'd0);   // REQ_SHIFT

        for (int v = 0; v < num_vecs; v++) begin
            logic signed [W-1:0] x [N];
            dummy = $fscanf(file_in, "%d %d %d %d %d %d %d %d\n",
                x_s[0], x_s[1], x_s[2], x_s[3], x_s[4], x_s[5], x_s[6], x_s[7]);
            for (int i = 0; i < N; i++) x[i] = W'(x_s[i]);

            load_vector(x);

            mmio_write(8'h00, 32'h01);  // CMD = SOFTMAX
            wait_for_done();

            // max / ln(S) / the (max+ln(S)) offset actually used for pass 4 --
            // bisects which internal stage first diverges from the model.
            $fwrite(stage_dbg_file, "%0d %0d %0d\n",
                $signed(dut.u_ctrl.reg_scalar_max), $signed(dut.u_ctrl.reg_scalar_log),
                $signed(dut.u_ctrl.reg_scalar_max + dut.u_ctrl.reg_scalar_log));

            // Softmax output is Q0.8 UNSIGNED (0..255, saturating) -- print
            // unsigned to match cpu_math_model.cpp's "softmax" mode.
            $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d\n",
                $unsigned(vpu_data_out[0]), $unsigned(vpu_data_out[1]),
                $unsigned(vpu_data_out[2]), $unsigned(vpu_data_out[3]),
                $unsigned(vpu_data_out[4]), $unsigned(vpu_data_out[5]),
                $unsigned(vpu_data_out[6]), $unsigned(vpu_data_out[7]));
        end

        $fclose(file_in);
        $fclose(file_out);
        $fclose(stage_dbg_file);
        $finish;
    end

endmodule
