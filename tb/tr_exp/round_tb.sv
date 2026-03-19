`timescale 1ns/1ps

module round_tb();

    // ========================================================================
    // MODE 1: 8-bit SoftMax/GELU (Q4.4) - Negative Only
    // ========================================================================
    logic signed [7:0] x_8b;
    logic              is_zero_8b;
    logic              is_ceil_8b;
    logic        [2:0] lut_idx_8b;
    
    round #(
        .WIDTH(8), 
        .FRAC_W(4), 
        .LUT_IDX_W(3)
    ) u_round_8b (
        .x(x_8b),
        .is_zero(is_zero_8b),
        .is_ceil(is_ceil_8b),
        .lut_idx(lut_idx_8b)
    );

    // ========================================================================
    // MODE 2: 12-bit LayerNorm (Q4.8) - Signed
    // ========================================================================
    logic signed [11:0] x_12b;
    logic               is_zero_12b;
    logic               is_ceil_12b;
    logic         [3:0] lut_idx_12b; // 4 bits to handle larger variance range
    
    round #(
        .WIDTH(12), 
        .FRAC_W(8), 
        .LUT_IDX_W(4)
    ) u_round_12b (
        .x(x_12b),
        .is_zero(is_zero_12b),
        .is_ceil(is_ceil_12b),
        .lut_idx(lut_idx_12b)
    );

    initial begin
        $display("==================================================");
        $display(" Testing Generalized tr_round.sv");
        $display("==================================================");

        $display("\n--- MODE 1: 8-bit Optimized (Negative Q4.4) ---");
        // Test -8.0 (8'b1000_0000)
        // Frac MSB is 0. Rounded is -8. ~(-8[2:0]) = ~(000) = 7.
        x_8b = 8'h80; #5; 
        $display("In: -8.0 (0x%0h) -> Zero:%b Ceil:%b Idx:%d (Expect 0, 0, 7)", x_8b, is_zero_8b, is_ceil_8b, lut_idx_8b);

        // Test -1.5 (8'b1110_1000)
        // Frac MSB is 1. Rounded is -2+1 = -1. ~(-1[2:0]) = ~(111) = 0.
        x_8b = 8'hE8; #5; 
        $display("In: -1.5 (0x%0h) -> Zero:%b Ceil:%b Idx:%d (Expect 0, 1, 0)", x_8b, is_zero_8b, is_ceil_8b, lut_idx_8b);

        // Test -0.5 (8'b1111_1000)
        // Frac MSB is 1. Rounded is -1+1 = 0. ~0 = 7. is_zero = 1.
        x_8b = 8'hF8; #5; 
        $display("In: -0.5 (0x%0h) -> Zero:%b Ceil:%b Idx:%d (Expect 1, 1, 7)", x_8b, is_zero_8b, is_ceil_8b, lut_idx_8b);


        $display("\n--- MODE 2: 12-bit Generic (Signed Q4.8) ---");
        // Test +2.5 (12'b0010_1000_0000)
        // Frac MSB is 1. Rounded is 2+1 = 3. Raw index should be 3.
        x_12b = 12'h280; #5; 
        $display("In: +2.5 (0x%0h) -> Zero:%b Ceil:%b Idx:%d (Expect 0, 1, 3)", x_12b, is_zero_12b, is_ceil_12b, lut_idx_12b);

        // Test -2.25 (12'b1101_1100_0000)
        // Frac MSB is 1. Rounded is -3+1 = -2. Raw index should be 14 (1110 in 4-bit).
        x_12b = 12'hDC0; #5; 
        $display("In: -2.25(0x%0h) -> Zero:%b Ceil:%b Idx:%d (Expect 0, 1, 14)", x_12b, is_zero_12b, is_ceil_12b, lut_idx_12b);

        $display("\n---> Rounding Test Complete!\n");
        $finish;
    end
endmodule