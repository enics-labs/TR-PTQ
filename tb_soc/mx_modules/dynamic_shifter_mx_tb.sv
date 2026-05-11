`timescale 1ns/1ps

module dynamic_shifter_mx_tb();

    localparam int N = 4;
    localparam int IN_W = 8;
    localparam int OUT_W = 16;

    // DUT Signals
    logic signed [IN_W-1:0]  data_in [N];
    logic signed [7:0]       shift_amount;
    logic                    shift_dir;
    logic signed [OUT_W-1:0] data_out [N];

    // Instantiate DUT
    dynamic_shifter_mx #(
        .N(N), .IN_W(IN_W), .OUT_W(OUT_W)
    ) dut (.*);

    initial begin
        $display("==================================================");
        $display("  Starting MX Dynamic Shifter Tests...            ");
        $display("==================================================");

        // TEST 1: Left Shift (Expansion into 16-bit)
        // data_in = {10, -5, 127, -128}
        // shift = 4 (Multiply by 16)
        // Expected = {160, -80, 2032, -2048}
        $display("\n[TEST 1] Left Shift (Expand by 4)");
        shift_dir    = 1'b1;
        shift_amount = 8'd4;
        data_in      = '{8'sd10, -8'sd5, 8'sd127, -8'sd128};
        #10;
        
        if (data_out[0] === 16'sd160 && data_out[1] === -16'sd80 &&
            data_out[2] === 16'sd2032 && data_out[3] === -16'sd2048)
            $display("  -> [PASS] Left shift and 16-bit expansion correct.");
        else
            $error("  -> [FAIL] Left shift failed. %p", data_out);

        // TEST 2: Right Shift (Compression/Division)
        // data_in (treated as 16-bit internal) = {64, -64, 0, 0}
        // We will pass them as 8-bit just for the TB flow, but shift_dir = 0
        // shift = 2 (Divide by 4)
        // Expected = {16, -16, 0, 0}
        $display("\n[TEST 2] Right Shift (Compress by 2)");
        shift_dir    = 1'b0;
        shift_amount = 8'd2;
        data_in      = '{8'sd64, -8'sd64, 8'sd0, 8'sd0};
        #10;

        if (data_out[0] === 16'sd16 && data_out[1] === -16'sd16)
            $display("  -> [PASS] Right shift and sign-extension correct.");
        else
            $error("  -> [FAIL] Right shift failed.");

        $display("\n==================================================");
        $finish;
    end
endmodule