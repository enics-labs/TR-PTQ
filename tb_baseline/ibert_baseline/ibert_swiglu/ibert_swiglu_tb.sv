`timescale 1ns/1ps

// File-I/O testbench of the standalone SwiGLU module ibert_swiglu: the same rows and the same "inputs.txt" / "hdl_out.txt" convention as
// tb_soc/tr_swiglu_int, so verify_block.py compares against the C++ golden model (ibert_swiglu). One row per line:
//   inputs.txt: "n g0 .. g(n-1) u0 .. u(n-1)";   hdl_out.txt: "e y0 .. y(n-1)"   (e = row gain)
// Per row, in 8-lane groups: pass 1 = SILU(gate) + MAX(up) per group, GAIN once, pass 2 = SILU(gate) + APPLY(up) per group; every command
// is followed by a wait for out_valid.
module ibert_swiglu_tb();

    localparam int N = 8, W = 8, OUT_W = 12;

    logic clk = 0, rst_n = 0;
    logic in_valid = 0;
    logic [1:0] cmd = 0;
    logic signed [W-1:0] x_in [N];
    logic out_valid;
    logic signed [OUT_W-1:0] y_out [N];
    logic [15:0] scale_o;

    ibert_swiglu #(.N(N), .W(W), .OUT_W(OUT_W)) dut (.*);

    always #5 clk = ~clk;

    // watchdog: a command that never completes ends the run with a message instead of hanging
    initial begin #400000000; $display("[TB WATCHDOG] timeout"); $finish; end

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
            logic signed [W-1:0] gate [];
            logic signed [W-1:0] up   [];
            logic signed [W-1:0] tg [N], tu [N];
            logic signed [OUT_W-1:0] y [];
            dummy = $fscanf(file_in, "%d", n);
            gate = new[n]; up = new[n]; y = new[n];
            for (int i = 0; i < n; i++) begin dummy = $fscanf(file_in, "%d", v_in); gate[i] = W'(v_in); end
            for (int i = 0; i < n; i++) begin dummy = $fscanf(file_in, "%d", v_in); up[i]   = W'(v_in); end
            dummy = $fscanf(file_in, "\n");

            for (int g = 0; g < n / N; g++) begin
                for (int i = 0; i < N; i++) begin tg[i] = gate[g*N + i]; tu[i] = up[g*N + i]; end
                issue(2'd0, tg);
                issue(2'd1, tu);
            end
            issue(2'd2, zero);
            $fwrite(file_out, "%0d", $signed(scale_o));
            for (int g = 0; g < n / N; g++) begin
                for (int i = 0; i < N; i++) begin tg[i] = gate[g*N + i]; tu[i] = up[g*N + i]; end
                issue(2'd0, tg);
                issue(2'd3, tu);
                for (int i = 0; i < N; i++) y[g*N + i] = y_out[i];
            end
            for (int i = 0; i < n; i++) $fwrite(file_out, " %0d", $signed(y[i]));
            $fwrite(file_out, "\n");
        end

        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

endmodule
