`timescale 1ns/1ps

// File-I/O testbench of the standalone RMSNorm modules (ibert_rmsnorm): the same rows and the same
// "inputs.txt" / "hdl_out.txt" convention as tb_soc/tr_rmsnorm_int, so verify_block.py compares against the C++ golden models
// (rmsnorm2 for tr_rmsnorm, rmsnorm_div for ibert_rmsnorm). One row per line:
//   inputs.txt: "log2n x0 .. x(N-1)";   hdl_out.txt: "scale y0 .. y(N-1)"   (scale = inv register of tr_rmsnorm, R of ibert_rmsnorm)
// Per row: ACC_FIRST / ACC per 8-lane tile, SCALE once, APPLY per tile; every command is followed by a wait for out_valid.
module ibert_rmsnorm_tb();

    localparam int N = 8, W = 8, OUT_W = 12;

    logic clk = 0, rst_n = 0;
    logic in_valid = 0;
    logic [1:0] cmd = 0;
    logic [3:0] log2n = 3;
    logic signed [W-1:0] x_in [N];
    logic out_valid;
    logic signed [OUT_W-1:0] y_out [N];
    logic [15:0] scale_o;

    ibert_rmsnorm #(.N(N), .W(W), .OUT_W(OUT_W)) dut (.*);

    always #5 clk = ~clk;

    // watchdog: a command that never completes ends the run with a message instead of hanging
    initial begin #200000000; $display("[TB WATCHDOG] timeout"); $finish; end

    task automatic issue(input logic [1:0] c, input logic signed [W-1:0] x [N]);
        @(posedge clk); #1;                          // inputs change after the edge (no race with the DUT's sampling)
        cmd = c; x_in = x; in_valid = 1;
        @(posedge clk); #1;
        in_valid = 0;
        while (!out_valid) @(posedge clk);
    endtask

    int file_in, file_out, num_rows, dummy, v_in, n;

    initial begin
        logic signed [W-1:0] zero [N];
        for (int i = 0; i < N; i++) begin x_in[i] = '0; zero[i] = '0; end
        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_rows);
        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        for (int r = 0; r < num_rows; r++) begin
            logic signed [W-1:0] row [];
            logic signed [W-1:0] tile [N];
            logic signed [OUT_W-1:0] y [];
            int l2;
            dummy = $fscanf(file_in, "%d", l2);
            n = 1 << l2;
            row = new[n]; y = new[n];
            for (int i = 0; i < n; i++) begin dummy = $fscanf(file_in, "%d", v_in); row[i] = W'(v_in); end
            dummy = $fscanf(file_in, "\n");
            log2n = 4'(l2);

            for (int t = 0; t < n / N; t++) begin
                for (int i = 0; i < N; i++) tile[i] = row[t*N + i];
                issue((t == 0) ? 2'd0 : 2'd1, tile);
            end
            issue(2'd2, zero);
            $fwrite(file_out, "%0d", scale_o);
            for (int t = 0; t < n / N; t++) begin
                for (int i = 0; i < N; i++) tile[i] = row[t*N + i];
                issue(2'd3, tile);
                for (int i = 0; i < N; i++) y[t*N + i] = y_out[i];
            end
            for (int i = 0; i < n; i++) $fwrite(file_out, " %0d", $signed(y[i]));
            $fwrite(file_out, "\n");
        end

        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

endmodule
