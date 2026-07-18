`timescale 1ns/1ps

// File-I/O testbench for the PRODUCTION GELU sequence: drives
// tr_soc_top_int through CMD=0x02 (GELU), the GL_P1..GL_P3 FSM in
// tr_soc_ctrl_int.sv sharing the tr_nonlinear_vpu backbone.  This is the
// module the firmware (tr_tensor.c's _xlr2_vpu(..., CMD_GELU, ...)) actually
// exercises -- NOT tr_gelu.sv (tb_soc/tr_gelu/), which is dead code outside
// tr_swiglu.sv (itself unused in the production ViT path), the same
// dead-code trap tr_rmsnorm.sv turned out to be.
//
// Isolates the GELU LUT/backbone itself (as opposed to tr_gelu_fused_int,
// which additionally exercises the accumulator->requantize passthrough):
// requantizer set to identity (mult=1, shift=0) so c_vec passes through
// unchanged, same convention as tr_rmsnorm_int_tb.sv.  An arbitrary 8-element
// GELU input x[0..7] is supplied as c_vec[0..3]=x[0..3] (lanes 0..M-1, via
// requantizer passthrough) and ext_sram_b[4..7]=x[4..7] (lanes M..N-1,
// software-supplied directly) -- see tr_soc_top_int.sv's vpu_sram_a_in mux.
//
// Matches verify_block.py's convention (inputs.txt / hdl_out.txt) and
// cpu_math_model.cpp's "gelu" mode (8 int8 in, 8 int8 out per line) --
// this REPLACES tr_gelu as the "gelu" block's RTL reference.
module tb_tr_gelu_int();

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

    // Capture the tr_exp_alu inputs/outputs for lane i at the exact cycle
    // GL_P2 writes scratch_b, before GL_P3 overwrites the combinational mux
    // state -- bisects whether bb_log (the wide value fed to tr_exp_alu)
    // matches the model's neg_ln, or whether a_idx/mantisa/is_zero diverge
    // given a matching bb_log.
    logic signed [7:0]  cap_bb_log   [N];
    logic [2:0]         cap_bb_aidx  [N];
    logic [7:0]         cap_bb_mant  [N];
    logic                cap_bb_zero  [N];
    always_ff @(posedge clk) begin
        if (dut.u_ctrl.write_scratch_b) begin
            cap_bb_log  <= dut.u_vpu.bb_log;
            cap_bb_aidx <= dut.u_vpu.bb_a_idx;
            cap_bb_mant <= dut.u_vpu.bb_mantisa;
            cap_bb_zero <= dut.u_vpu.bb_is_zero;
        end
    end

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
    // the requantizer, exactly as tr_rmsnorm_int_tb.sv's load_vector does.
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

    int file_in, file_out, stage_dbg_file, exp_dbg_file, num_vecs, dummy;
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
        exp_dbg_file = $fopen("exp_dbg.txt", "w");
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

            mmio_write(8'h00, 32'h02);  // CMD = GELU
            wait_for_done();

            // E (post GL_P1, scratch_a) and recip (post GL_P2, scratch_b) --
            // bisects which internal stage first diverges from the model.
            $fwrite(stage_dbg_file, "%0d %0d %0d %0d %0d %0d %0d %0d  %0d %0d %0d %0d %0d %0d %0d %0d\n",
                $signed(dut.ctrl_scratch_a[0]), $signed(dut.ctrl_scratch_a[1]),
                $signed(dut.ctrl_scratch_a[2]), $signed(dut.ctrl_scratch_a[3]),
                $signed(dut.ctrl_scratch_a[4]), $signed(dut.ctrl_scratch_a[5]),
                $signed(dut.ctrl_scratch_a[6]), $signed(dut.ctrl_scratch_a[7]),
                $signed(dut.ctrl_scratch_b[0]), $signed(dut.ctrl_scratch_b[1]),
                $signed(dut.ctrl_scratch_b[2]), $signed(dut.ctrl_scratch_b[3]),
                $signed(dut.ctrl_scratch_b[4]), $signed(dut.ctrl_scratch_b[5]),
                $signed(dut.ctrl_scratch_b[6]), $signed(dut.ctrl_scratch_b[7]));

            $fwrite(exp_dbg_file, "%0d %0d %0d %0d  %0d %0d %0d %0d\n",
                cap_bb_log[0], cap_bb_aidx[0], cap_bb_mant[0], cap_bb_zero[0],
                cap_bb_log[1], cap_bb_aidx[1], cap_bb_mant[1], cap_bb_zero[1]);

            $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d\n",
                $signed(vpu_data_out[0]), $signed(vpu_data_out[1]),
                $signed(vpu_data_out[2]), $signed(vpu_data_out[3]),
                $signed(vpu_data_out[4]), $signed(vpu_data_out[5]),
                $signed(vpu_data_out[6]), $signed(vpu_data_out[7]));
        end

        $fclose(file_in);
        $fclose(file_out);
        $fclose(stage_dbg_file);
        $fclose(exp_dbg_file);
        $finish;
    end

endmodule
