`timescale 1ns/1ps

module tb_tr_gelu();

    localparam int N    = 8;
    localparam int W    = 8;
    localparam int ACC_W = 32;

    logic                clk = 0;
    logic                rst_n = 0;
    logic                valid_in = 0;
    logic [1:0]          mode = '0;
    logic signed [W-1:0] x_in   [N];
    logic signed [W-1:0] aux_in [N];
    logic                valid_out;
    logic signed [W-1:0] y_out  [N];

    tr_gelu #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) dut (.*);

    always #5 clk = ~clk;

    // Assert valid_in for one cycle, then wait for valid_out.
    task automatic send_and_wait();
        @(posedge clk); valid_in = 1'b1;
        @(posedge clk); valid_in = 1'b0;
        while (!valid_out) @(posedge clk);
    endtask

    int file_in, file_out, num_vecs, dummy, stimulus[N];

    initial begin
        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_vecs);

        for (int i = 0; i < N; i++) begin x_in[i] = '0; aux_in[i] = '0; end
        #20; rst_n = 1;

        for (int v = 0; v < num_vecs; v++) begin
            logic signed [W-1:0] x_saved  [N];
            logic signed [W-1:0] ctrl_E   [N];
            logic signed [W-1:0] ctrl_recip[N];

            dummy = $fscanf(file_in, "%d %d %d %d %d %d %d %d\n",
                stimulus[0], stimulus[1], stimulus[2], stimulus[3],
                stimulus[4], stimulus[5], stimulus[6], stimulus[7]);
            for (int j = 0; j < N; j++) begin
                x_saved[j] = W'(stimulus[j]);
                x_in[j]    = x_saved[j];
                aux_in[j]  = '0;
            end

            // ------ Pass 0: E = exp(-alpha|x|) ------
            mode = 2'b00;
            send_and_wait();
            for (int j = 0; j < N; j++) ctrl_E[j] = y_out[j];

            // ------ Pass 1: recip = 1/(1+E) ------
            mode = 2'b01;
            for (int j = 0; j < N; j++) begin x_in[j] = ctrl_E[j]; aux_in[j] = '0; end
            send_and_wait();
            for (int j = 0; j < N; j++) ctrl_recip[j] = y_out[j];

            // ------ Pass 2: y = x * sigma(x) ------
            mode = 2'b10;
            for (int j = 0; j < N; j++) begin x_in[j] = x_saved[j]; aux_in[j] = ctrl_recip[j]; end
            send_and_wait();

            $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d\n",
                $signed(y_out[0]), $signed(y_out[1]), $signed(y_out[2]), $signed(y_out[3]),
                $signed(y_out[4]), $signed(y_out[5]), $signed(y_out[6]), $signed(y_out[7]));
        end

        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

endmodule
