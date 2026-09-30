`timescale 1ns/1ps

module ibert_gelu_tb();

    localparam int N      = 8;
    localparam int W      = 8;
    localparam int FRAC_W = 4;

    logic clk, rst_n;
    logic valid_in, valid_out;
    logic signed [W-1:0] x_in [N];
    logic signed [W-1:0] y_out [N];

    ibert_gelu #(.N(N), .W(W), .FRAC_W(FRAC_W)) dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .x_in(x_in), .valid_out(valid_out), .y_out(y_out)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    int file_in, file_out, n, dummy;
    byte tag;
    int xv [N];

    initial begin
        rst_n = 0; valid_in = 0;
        for (int i = 0; i < N; i++) x_in[i] = '0;
        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        file_in  = $fopen("inputs.txt", "r");
        file_out = $fopen("hdl_out.txt", "w");
        if (!file_in || !file_out) begin
            $display("[ERROR] could not open IO files");
            $finish;
        end
        dummy = $fscanf(file_in, "%d\n", n);
        for (int i = 0; i < n; i++) begin
            dummy = $fscanf(file_in, "%c", tag);
            if (tag == "E") begin
                dummy = $fscanf(file_in, "%d\n", xv[0]);
                for (int lane = 0; lane < N; lane++) x_in[lane] = W'(xv[0]);
            end else begin
                for (int lane = 0; lane < N; lane++) dummy = $fscanf(file_in, "%d", xv[lane]);
                dummy = $fscanf(file_in, "\n");
                for (int lane = 0; lane < N; lane++) x_in[lane] = W'(xv[lane]);
            end
            valid_in = 1'b1;
            @(posedge clk);
            valid_in = 1'b0;
            while (!valid_out) @(posedge clk);
            if (tag == "E") begin
                $fwrite(file_out, "%0d\n", $signed(y_out[0]));
            end else begin
                for (int lane = 0; lane < N; lane++)
                    $fwrite(file_out, "%0d%s", $signed(y_out[lane]), (lane == N-1) ? "\n" : " ");
            end
            @(posedge clk);
        end
        $fclose(file_in);
        $fclose(file_out);
        $display("[GELU TB] evaluated %0d vectors", n);
        $finish;
    end

endmodule
