`timescale 1ns/1ps

// File-I/O testbench for the PRODUCTION two-pass SwiGLU: drives tr_soc_top_int through the host-sequenced
// command set of tr_soc_ctrl_int.sv (cmd 5 silu group, 6 max group, 0xB row gain, 0xC apply group):
//   pass 1, per 8-lane group: stage gate x, CMD=5;  stage up u, CMD=6
//   once per row:             CMD=0xB, then read the gain e at ADDR_SW_E (0x1C)
//   pass 2, per group:        stage x, CMD=5;  stage u, CMD=0xC  -> int8 result
// One row (n elements) per input line: inputs.txt "n g0..g(n-1) u0..u(n-1)"; hdl_out.txt "e y0..y(n-1)".
// Golden: cpu_math_model.cpp "swiglu" (sw_* in tr_math_model.hpp), bit-exact to tools/models/vpu_ops.py::swiglu.
//
// The VPU's 8-wide input is composed of two paths (see tr_soc_top_int.sv):
//   lanes 0..M-1 (0..3): requantizer output, fed via c_vec passthrough
//                        (a_mat=0, b_vec=0, mult=1, shift=0 => output = c_vec)
//   lanes M..N-1 (4..7): ext_sram_b, driven directly
// So an arbitrary 8-element RMSNorm input x[0..7] is supplied as
// c_vec[0..3]=x[0..3], ext_sram_b[4..7]=x[4..7].
//
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

    int file_in, file_out, num_rows, dummy;
    int n, v_in;

    task automatic stage_group(input logic signed [W-1:0] row [], input int base);
        logic signed [W-1:0] tile [N];
        for (int i = 0; i < N; i++) tile[i] = row[base + i];
        load_vector(tile);
    endtask

    initial begin
        rst_n = 0; mmio_wen = 0; dot_in_valid = 0; clear_acc = 1;
        for (int i = 0; i < N; i++) begin b_vec[i] = '0; ext_sram_b[i] = '0; end
        for (int m = 0; m < M; m++) begin
            c_vec[m] = 0;
            for (int i = 0; i < N; i++) a_mat[m][i] = '0;
        end

        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_rows);

        #22 rst_n = 1;

        // Requantizer pass-through: mult=1, shift=0 => c_vec passes unchanged.
        mmio_write(8'h08, 32'd1);   // REQ_MULT
        mmio_write(8'h0C, 32'd0);   // REQ_SHIFT

        for (int r = 0; r < num_rows; r++) begin
            logic signed [W-1:0] gate [];
            logic signed [W-1:0] up   [];
            logic signed [W-1:0] y    [];

            dummy = $fscanf(file_in, "%d", n);
            gate = new[n]; up = new[n]; y = new[n];
            for (int i = 0; i < n; i++) begin dummy = $fscanf(file_in, "%d", v_in); gate[i] = W'(v_in); end
            for (int i = 0; i < n; i++) begin dummy = $fscanf(file_in, "%d", v_in); up[i]   = W'(v_in); end
            dummy = $fscanf(file_in, "\n");

            // Pass 1: running max of |silu * up|
            for (int g = 0; g < n / N; g++) begin
                stage_group(gate, g*N); mmio_write(8'h00, 32'h05); wait_for_done();
                stage_group(up,   g*N); mmio_write(8'h00, 32'h06); wait_for_done();
            end

            // Row gain
            mmio_write(8'h00, 32'h0B); wait_for_done();
            begin
                logic signed [31:0] e_rd;
                @(posedge clk); mmio_addr = 8'h1C; mmio_wen = 0; #1 e_rd = mmio_rdata;
                $fwrite(file_out, "%0d", $signed(e_rd));
            end

            // Pass 2: requantize with the row gain
            for (int g = 0; g < n / N; g++) begin
                stage_group(gate, g*N); mmio_write(8'h00, 32'h05); wait_for_done();
                stage_group(up,   g*N); mmio_write(8'h00, 32'h0C); wait_for_done();
                for (int i = 0; i < N; i++) y[g*N + i] = vpu_data_out[i];
            end

            for (int i = 0; i < n; i++) $fwrite(file_out, " %0d", $signed(y[i]));
            $fwrite(file_out, "\n");
        end

        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

endmodule
