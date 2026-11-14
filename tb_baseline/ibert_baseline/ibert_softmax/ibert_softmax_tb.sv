`timescale 1ns/1ps

// File-I/O testbench of the standalone softmax tile ibert_softmax: the same tiles and the same "inputs.txt" / "hdl_out.txt" convention as
// tb_soc/tr_softmax_int, so verify_block.py compares against the C++ golden model (ibert_softmax).
//   inputs.txt: first line the number of tiles, then "x0 .. x7" per tile;   hdl_out.txt: "p0 .. p7" per tile
module ibert_softmax_tb();

    localparam int N = 8, W = 8;

    logic clk = 0, rst_n = 0;
    logic in_valid = 0;
    logic signed [W-1:0] x_in [N];
    logic out_valid;
    logic [W-1:0] p_out [N];

    ibert_softmax #(.N(N), .W(W)) dut (.*);

    always #5 clk = ~clk;

    // watchdog: a tile that never completes ends the run with a message instead of hanging
    initial begin #400000000; $display("[TB WATCHDOG] timeout"); $finish; end

    int file_in, file_out, num_vecs, dummy, v;

    initial begin
        for (int i = 0; i < N; i++) x_in[i] = '0;
        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_vecs);
        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        for (int t = 0; t < num_vecs; t++) begin
            for (int i = 0; i < N; i++) begin dummy = $fscanf(file_in, "%d", v); x_in[i] = W'(v); end
            dummy = $fscanf(file_in, "\n");

            @(posedge clk); #1;                      // inputs change after the edge (no race with the DUT's sampling)
            in_valid = 1;
            @(posedge clk); #1;
            in_valid = 0;
            while (!out_valid) @(posedge clk);
            #1;
            $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d\n", $unsigned(p_out[0]), $unsigned(p_out[1]), $unsigned(p_out[2]), $unsigned(p_out[3]), $unsigned(p_out[4]), $unsigned(p_out[5]), $unsigned(p_out[6]), $unsigned(p_out[7]));
        end

        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

endmodule
