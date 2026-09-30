`timescale 1ns/1ps

// File-I/O testbench for tr_rmsnorm, matching the verify_block.py
// convention (inputs.txt / hdl_out.txt) used by tr_gelu_tb.sv/
// tr_softmax_tb.sv. Verifies against cpu_math_model.cpp's "rmsnorm"/
// "rmsnorm_baseline" mode (rmsnorm_hw_model): Gamma is implicitly 1.0
// (Q4.4 code 16) for every lane, matching that model -- no gamma column in
// inputs.txt, same 8-int8-per-line format as "gelu"/"rmsnorm".
//
// Drives the SAME mode 00->01->10->11 pass sequence already exercised by
// the directed regression test (tr_rmsnorm_directed_tb.sv, kept separately
// for its hardcoded sign-guard edge cases) -- see that file/tr_rmsnorm.sv's
// own header comment for what each pass does.
module tb_tr_rmsnorm();

    localparam int N     = 8;
    localparam int W     = 8;
    localparam int ACC_W = 32;

    logic                    clk = 0;
    logic                    rst_n = 0;
    logic                    valid_in = 0;
    logic [1:0]               mode = '0;
    logic signed [W-1:0]      x_in   [N];
    logic signed [W-1:0]      aux_in [N];
    logic signed [ACC_W-1:0]  sum_in = '0;
    logic signed [W-1:0]      offset_in = '0;

    logic                     valid_out;
    logic signed [W-1:0]      y_out   [N];
    logic signed [ACC_W-1:0]  sum_out;
    logic signed [W-1:0]      ln_out;

    // 0.5*ln(N) in Q4.4, N=8 -- same constant tr_soc_ctrl_int.sv's
    // CONST_LN_SQRT_N and the directed testbench use.
    localparam logic signed [W-1:0] CONST_LN_SQRT_N = 8'd17;
    localparam logic signed [W-1:0] GAMMA_ONE        = 8'd16;   // 1.0 in Q4.4

    tr_rmsnorm #(.N(N), .W(W), .FRAC_W(4), .ACC_W(ACC_W)) dut (.*);

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
            logic signed [W-1:0]     x_saved  [N];
            logic signed [ACC_W-1:0] ctrl_sum;
            logic signed [W-1:0]     ctrl_V   [N];
            logic signed [W-1:0]     ctrl_log_offset;
            logic signed [W-1:0]     ctrl_inv_rms;

            dummy = $fscanf(file_in, "%d %d %d %d %d %d %d %d\n",
                stimulus[0], stimulus[1], stimulus[2], stimulus[3],
                stimulus[4], stimulus[5], stimulus[6], stimulus[7]);
            for (int j = 0; j < N; j++) x_saved[j] = W'(stimulus[j]);

            // ------ Pass 1 (mode 00): S = sum(x*x) ------
            mode = 2'b00;
            for (int j = 0; j < N; j++) begin x_in[j] = x_saved[j]; aux_in[j] = x_saved[j]; end
            send_and_wait();
            ctrl_sum = sum_out;

            // ------ Pass 2 (mode 01): V = x*Gamma, log_offset = -0.5*ln(S) + ln(sqrt(N)) ------
            mode = 2'b01;
            for (int j = 0; j < N; j++) begin x_in[j] = x_saved[j]; aux_in[j] = GAMMA_ONE; end
            sum_in = ctrl_sum;
            send_and_wait();
            for (int j = 0; j < N; j++) ctrl_V[j] = y_out[j];
            ctrl_log_offset = ln_out + CONST_LN_SQRT_N;

            // ------ Pass 3 (mode 10): InvRMS = exp(log_offset), sign-guarded ------
            mode = 2'b10;
            offset_in = ctrl_log_offset;
            send_and_wait();
            ctrl_inv_rms = y_out[0];

            // ------ Pass 4 (mode 11): y = V * InvRMS ------
            mode = 2'b11;
            for (int j = 0; j < N; j++) begin x_in[j] = ctrl_V[j]; aux_in[j] = ctrl_inv_rms; end
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
