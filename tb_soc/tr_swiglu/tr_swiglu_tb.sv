`timescale 1ns/1ps

module tb_tr_swiglu();

    localparam int N     = 8;
    localparam int W     = 8;
    localparam int ACC_W = 32;

    logic                clk = 0;
    logic                rst_n = 0;
    logic                valid_in = 0;
    logic [1:0]          mode = '0;
    logic signed [W-1:0] x_in   [N];
    logic signed [W-1:0] aux_in [N];
    logic                valid_out;
    logic signed [W-1:0] y_out  [N];

    tr_swiglu #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) dut (.*);

    always #5 clk = ~clk;

    task automatic send_and_wait();
        @(posedge clk); valid_in = 1'b1;
        @(posedge clk); valid_in = 1'b0;
        while (!valid_out) @(posedge clk);
    endtask

    int file_in, file_out, num_vecs, dummy;
    int stimulus_x[N], stimulus_g[N];

    initial begin
        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_vecs);

        for (int i = 0; i < N; i++) begin x_in[i] = '0; aux_in[i] = '0; end
        #20; rst_n = 1;

        for (int v = 0; v < num_vecs; v++) begin
            logic signed [W-1:0] x_saved [N];
            logic signed [W-1:0] g_saved [N];
            logic signed [W-1:0] ctrl_E  [N];
            logic signed [W-1:0] ctrl_recip [N];
            logic signed [W-1:0] ctrl_silu  [N];

            // Each input line: x[0..7]  g[0..7]
            dummy = $fscanf(file_in,
                "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                stimulus_x[0], stimulus_x[1], stimulus_x[2], stimulus_x[3],
                stimulus_x[4], stimulus_x[5], stimulus_x[6], stimulus_x[7],
                stimulus_g[0], stimulus_g[1], stimulus_g[2], stimulus_g[3],
                stimulus_g[4], stimulus_g[5], stimulus_g[6], stimulus_g[7]);

            for (int j = 0; j < N; j++) begin
                x_saved[j] = W'(stimulus_x[j]);
                g_saved[j] = W'(stimulus_g[j]);
            end

            // ------ Pass 0: E = exp(-alpha|x|) ------
            mode = 2'b00;
            for (int j = 0; j < N; j++) begin x_in[j] = x_saved[j]; aux_in[j] = '0; end
            send_and_wait();
            for (int j = 0; j < N; j++) ctrl_E[j] = y_out[j];

            // ------ Pass 1: recip = 1/(1+E) = sigma(x) ------
            mode = 2'b01;
            for (int j = 0; j < N; j++) begin x_in[j] = ctrl_E[j]; aux_in[j] = '0; end
            send_and_wait();
            for (int j = 0; j < N; j++) ctrl_recip[j] = y_out[j];

            // ------ Pass 2: silu = x * sigma(x) ------
            mode = 2'b10;
            for (int j = 0; j < N; j++) begin
                x_in[j]   = x_saved[j];
                aux_in[j] = ctrl_recip[j];
            end
            send_and_wait();
            for (int j = 0; j < N; j++) ctrl_silu[j] = y_out[j];

            // ------ Pass 3: y = silu * g ------
            mode = 2'b11;
            for (int j = 0; j < N; j++) begin
                x_in[j]   = ctrl_silu[j];
                aux_in[j] = g_saved[j];
            end
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
