`timescale 1ns/1ps

module quadratic_divider_tb();

    // ========================================================================
    // MODE 1: 8-bit Optimized (4-bit fractional Delta)
    // ========================================================================
    logic signed [3:0] delta_8b;
    logic        [3:0] quad_8b;
    
    quadratic_divider #(
        .WIDTH(8), 
        .FRAC_W(4)
    ) u_quad_8b (
        .delta(delta_8b),
        .quad_out(quad_8b)
    );

    // ========================================================================
    // MODE 2: 12-bit Generic (8-bit fractional Delta)
    // ========================================================================
    logic signed [7:0] delta_12b;
    logic        [7:0] quad_12b;
    
    quadratic_divider #(
        .WIDTH(12), 
        .FRAC_W(8)
    ) u_quad_12b (
        .delta(delta_12b),
        .quad_out(quad_12b)
    );

    initial begin
        $display("==================================================");
        $display(" Testing Generalized tr_quad_div.sv");
        $display("==================================================");

        $display("\n--- MODE 1: 8-bit K-Map Optimized ---");
        // Test 0
        delta_8b = 4'sd0; #5; 
        $display("Delta: %4d -> Quad: %d (Expect 0)", delta_8b, quad_8b);
        
        // Test -7 (Should trigger y[0] = 1)
        delta_8b = -4'sd7; #5; 
        $display("Delta: %4d -> Quad: %d (Expect 1)", delta_8b, quad_8b);
        
        // Test -8 (Should trigger y[1] = 1, meaning 2)
        delta_8b = -4'sd8; #5; 
        $display("Delta: %4d -> Quad: %d (Expect 2)", delta_8b, quad_8b);


        $display("\n--- MODE 2: 12-bit Generic Multiplier ---");
        // We scale the test values up for 8 fractional bits (x16) to test equivalence
        
        // Test 0
        delta_12b = 8'sd0; #5; 
        $display("Delta: %4d -> Quad: %d (Expect 0)", delta_12b, quad_12b);
        
        // Test -0.4375 (equivalent to -7 in Q.4 -> -112 in Q.8)
        // Math: (-112 * -112) >> 9 = 12544 >> 9 = 24.
        delta_12b = -8'sd112; #5; 
        $display("Delta: %4d -> Quad: %d (Expect 24)", delta_12b, quad_12b);
        
        // Test -0.5 (equivalent to -8 in Q.4 -> -128 in Q.8)
        // Math: (-128 * -128) >> 9 = 16384 >> 9 = 32.
        delta_12b = -8'sd128; #5; 
        $display("Delta: %4d -> Quad: %d (Expect 32)", delta_12b, quad_12b);

        $display("\n---> Quadratic Divider Test Complete!\n");
        $finish;
    end
endmodule