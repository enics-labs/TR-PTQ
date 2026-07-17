`timescale 1ns/1ps

// File-I/O testbench for the PRODUCTION RMSNorm sequence: drives
// tr_soc_top_int through CMD=0x03 (RMSNORM), the RM_P1..RM_P4 FSM in
// tr_soc_ctrl_int.sv sharing the tr_nonlinear_vpu backbone.  This is the
// module the firmware (tr_rmsnorm -> _xlr2_vpu) actually exercises.
//
// (tr_rmsnorm.sv / tr_rmsnorm_tb.sv are NOT this -- confirmed by grep that
// module is never instantiated in production RTL; do not use it as a
// reference.)
//
// The VPU's 8-wide input is composed of two paths (see tr_soc_top_int.sv):
//   lanes 0..M-1 (0..3): requantizer output, fed via c_vec passthrough
//                        (a_mat=0, b_vec=0, mult=1, shift=0 => output = c_vec)
//   lanes M..N-1 (4..7): ext_sram_b, driven directly
// So an arbitrary 8-element RMSNorm input x[0..7] is supplied as
// c_vec[0..3]=x[0..3], ext_sram_b[4..7]=x[4..7].
//
// Matches verify_block.py's convention (inputs.txt / hdl_out.txt) and
// cpu_math_model.cpp's "rmsnorm" mode (8 int8 in, 8 int8 out per line).
module tb_tr_rmsnorm_int();

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
    // the requantizer, exactly as tr_soc_top_int_tb.sv's load_mac_vector does.
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

    int file_in, file_out, num_vecs, dummy;
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

            mmio_write(8'h00, 32'h03);  // CMD = RMSNORM
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