`timescale 1ns/1ps

// File-I/O testbench for requantize_engine_int, matching the verify_block.py
// convention (inputs.txt / hdl_out.txt) used by tr_exp/tr_gelu/etc.
// Verifies against cpu_math_model.cpp's "quant" mode: Saturate((acc*mult +
// bias) >>> shift), bias = shift>0 ? 1<<(shift-1) : 0.  N=4 to match the
// existing requantize_engine_int_tb.sv parameterization.
module tb_tr_quant();

    localparam int N       = 4;
    localparam int ACC_W   = 32;
    localparam int MUL_W   = 32;
    localparam int SHIFT_W = 6;
    localparam int OUT_W   = 8;

    logic clk = 0, rst_n = 0;
    logic in_valid, in_ready;
    logic signed [ACC_W-1:0] acc_in [N];
    logic signed [MUL_W-1:0] multiplier;
    logic [SHIFT_W-1:0]      shift;
    logic out_valid, out_ready;
    logic signed [OUT_W-1:0] out_vec [N];

    requantize_engine_int #(
        .N(N), .ACC_W(ACC_W), .MUL_W(MUL_W), .SHIFT_W(SHIFT_W), .OUT_W(OUT_W)
    ) dut (.*);

    always #5 clk = ~clk;

    int file_in, file_out, num_vecs, dummy;
    longint acc_s[N]; int mult_s, shift_s;

    initial begin
        file_in  = $fopen("inputs.txt",  "r");
        file_out = $fopen("hdl_out.txt", "w");
        dummy    = $fscanf(file_in, "%0d\n", num_vecs);

        in_valid = 0; out_ready = 1;
        for (int i = 0; i < N; i++) acc_in[i] = '0;
        multiplier = '0; shift = '0;
        #20; rst_n = 1;
        @(posedge clk);

        for (int v = 0; v < num_vecs; v++) begin
            dummy = $fscanf(file_in, "%d %d %d %d %d %d\n",
                acc_s[0], acc_s[1], acc_s[2], acc_s[3], mult_s, shift_s);

            for (int i = 0; i < N; i++) acc_in[i] = ACC_W'(acc_s[i]);
            multiplier = MUL_W'(mult_s);
            shift      = SHIFT_W'(shift_s);

            @(negedge clk); in_valid = 1;
            @(negedge clk); in_valid = 0;
            while (!out_valid) @(posedge clk);

            $fwrite(file_out, "%0d %0d %0d %0d\n",
                $signed(out_vec[0]), $signed(out_vec[1]),
                $signed(out_vec[2]), $signed(out_vec[3]));

            @(posedge clk);
        end

        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

endmodule