`timescale 1ns/1ps

module tr_softmax_tb();
    localparam int N = 8, W = 8, ACC_W = 32;

    logic clk = 0;
    logic rst_n = 0;
    always #5 clk = ~clk;

    // DUT ports
    logic                    valid_in  = 0;
    logic                    valid_out;
    logic [2:0]              mode      = '0;
    logic signed [W-1:0]     x_in      [N];
    logic signed [W-1:0]     offset_in = '0;
    logic signed [ACC_W-1:0] sum_in    = '0;
    logic signed [W-1:0]     y_out     [N];
    logic signed [ACC_W-1:0] sum_out;

    tr_softmax #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) uut (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .valid_out (valid_out),
        .mode      (mode),
        .x_in      (x_in),
        .offset_in (offset_in),
        .sum_in    (sum_in),
        .y_out     (y_out),
        .sum_out   (sum_out)
    );

    int file_in, file_out, num_vecs, dummy, stimulus[N];

    // Assert valid_in for one cycle then wait until valid_out fires.
    // Not suitable for the combinational pass 3 — that is inlined below.
    task automatic send_and_wait();
        @(posedge clk); valid_in = 1'b1;
        @(posedge clk); valid_in = 1'b0;
        while (!valid_out) @(posedge clk);
    endtask

    initial begin
        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy = $fscanf(file_in, "%0d\n", num_vecs);

        for (int i = 0; i < N; i++) x_in[i] = '0;
        #20; rst_n = 1;

        for (int v = 0; v < num_vecs; v++) begin
            // Per-vector intermediate results
            logic signed [W-1:0]     max_val;
            logic signed [ACC_W-1:0] S;
            logic signed [W-1:0]     ln_S;
            logic signed [W-1:0]     exp_vals [N];
            logic signed [W-1:0]     inv_S;

            dummy = $fscanf(file_in, "%d %d %d %d %d %d %d %d\n",
                stimulus[0], stimulus[1], stimulus[2], stimulus[3],
                stimulus[4], stimulus[5], stimulus[6], stimulus[7]);
            for (int j = 0; j < N; j++) x_in[j] = W'(stimulus[j]);

            // ------ Pass 1: max(X) ------
            mode = 3'b000; offset_in = '0;
            send_and_wait();
            max_val = y_out[0];

            // ------ Pass 2: exp(xi − max) DOT → S = Σ exp(xi−max) ------
            mode = 3'b001; offset_in = max_val;
            send_and_wait();
            S = sum_out;

            // ------ Pass 3: ln(S) — combinational, no pipeline wait ------
            mode = 3'b010; sum_in = S;
            @(posedge clk); valid_in = 1'b1;
            @(posedge clk); ln_S = y_out[0]; valid_in = 1'b0;

            // ------ Pass 4: exp(xi − max) ELEMWISE → individual exp values ------
            mode = 3'b011; offset_in = max_val;
            for (int j = 0; j < N; j++) x_in[j] = W'(stimulus[j]);
            send_and_wait();
            for (int j = 0; j < N; j++) exp_vals[j] = y_out[j];

            // ------ Pass 5: exp(0 − ln(S)) = 1/S ------
            // Feed x_in=0, offset_in=ln(S) so x_sub = −ln(S) through the exp ALU
            mode = 3'b100; offset_in = ln_S;
            for (int j = 0; j < N; j++) x_in[j] = '0;
            send_and_wait();
            inv_S = y_out[0];

            // ------ Pass 6: exp(xi−max) × 1/S → final probabilities ------
            mode = 3'b101; offset_in = inv_S;
            for (int j = 0; j < N; j++) x_in[j] = exp_vals[j];
            send_and_wait();

            // Pass 6 output is now an UNSIGNED Q0.8 probability (0..255) --
            // print unsigned to match cpu_math_model.cpp's golden output.
            $fwrite(file_out, "%0d %0d %0d %0d %0d %0d %0d %0d\n",
                $unsigned(y_out[0]), $unsigned(y_out[1]), $unsigned(y_out[2]), $unsigned(y_out[3]),
                $unsigned(y_out[4]), $unsigned(y_out[5]), $unsigned(y_out[6]), $unsigned(y_out[7]));

            // Debug print for every vector — shows all pass intermediates
            if (v == 20 || v == 24) begin
                $display("--- Vec %0d ---", v+1);
                $display("  Input   : %0d %0d %0d %0d %0d %0d %0d %0d",
                    stimulus[0], stimulus[1], stimulus[2], stimulus[3],
                    stimulus[4], stimulus[5], stimulus[6], stimulus[7]);
                $display("  P1 max  : %0d", $signed(max_val));
                $display("  P2 S    : %0d  [23:8]=%0d", S, S[23:8]);
                $display("  P3 ln_S : %0d", $signed(ln_S));
                // Passes 4/5/6 are UNSIGNED Q0.8 magnitudes now -- print unsigned.
                $display("  P4 exp  : %0d %0d %0d %0d %0d %0d %0d %0d",
                    $unsigned(exp_vals[0]), $unsigned(exp_vals[1]), $unsigned(exp_vals[2]), $unsigned(exp_vals[3]),
                    $unsigned(exp_vals[4]), $unsigned(exp_vals[5]), $unsigned(exp_vals[6]), $unsigned(exp_vals[7]));
                $display("  P5 inv_S: %0d", $unsigned(inv_S));
                $display("  P6 out  : %0d %0d %0d %0d %0d %0d %0d %0d",
                    $unsigned(y_out[0]), $unsigned(y_out[1]), $unsigned(y_out[2]), $unsigned(y_out[3]),
                    $unsigned(y_out[4]), $unsigned(y_out[5]), $unsigned(y_out[6]), $unsigned(y_out[7]));
            end
        end

        $fclose(file_in); $fclose(file_out);
        $finish;
    end
endmodule
