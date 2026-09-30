`timescale 1ns/1ps

module ibert_divider_tb();

    localparam int WIDTH = 24;
    localparam int NUM_VECS = 2000;

    logic clk, rst_n;
    logic valid_in;
    logic [WIDTH-1:0] dividend, divisor;
    logic busy, valid_out;
    logic [WIDTH-1:0] quotient, remainder;

    ibert_divider #(.WIDTH(WIDTH)) dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .dividend(dividend), .divisor(divisor),
        .busy(busy), .valid_out(valid_out),
        .quotient(quotient), .remainder(remainder)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    int file_in, file_out, n, dummy;
    int unsigned dvd, dvs;

    initial begin
        rst_n = 0; valid_in = 0; dividend = '0; divisor = '0;
        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        file_in = $fopen("inputs.txt", "r");
        file_out = $fopen("hdl_out.txt", "w");
        if (!file_in || !file_out) begin
            $display("[ERROR] could not open IO files");
            $finish;
        end
        dummy = $fscanf(file_in, "%d\n", n);
        for (int i = 0; i < n; i++) begin
            dummy = $fscanf(file_in, "%d %d\n", dvd, dvs);
            while (busy) @(posedge clk);
            dividend = dvd[WIDTH-1:0];
            divisor  = dvs[WIDTH-1:0];
            valid_in = 1'b1;
            @(posedge clk);
            valid_in = 1'b0;
            while (!valid_out) @(posedge clk);
            $fwrite(file_out, "%0d %0d\n", quotient, remainder);
            @(posedge clk);
        end
        $fclose(file_in);
        $fclose(file_out);
        $display("[DIVIDER TB] evaluated %0d vectors", n);
        $finish;
    end

endmodule
