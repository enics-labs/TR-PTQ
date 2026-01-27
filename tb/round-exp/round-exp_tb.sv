`timescale 1ns/1ps

module q4_4_round_neg_tb();

    reg  signed [7:0] x;
    wire              is_zero;
    wire        [2:0] fliped_rounded_int;
    logic       [7:0]         mantisa;
    logic       [7:0]         e_a;
    
    parameter FRAC = 4;
    parameter ITER = 2;

    tr_exp #(
        .FRAC(FRAC),
        .ITER(ITER)
    ) uut (
        .x(x),      // Q4
        .e_a(e_a),      // Q4
        .mantisa(mantisa),
        .is_zero(is_zero)
    );

    import "DPI-C" function int dpi_real_to_qmk(
        input real real_val,
        input int  M,
        input int  K
    );
    import "DPI-C" function real dpi_qmk_to_real(
        input int fixed_val,
        input int K
    );

    initial begin
        $display("---------------------------------------------------------");
        $display("  Input (Q4.4) | Dec Value | Rounded Int | LUT Index | mantisa    |    mantisa    | Zero");
        $display("---------------------------------------------------------");

        // 1. Test Zero
        x = 8'sh00; #10; display_vals();

        // 2. Test Small Negatives (Rounding to 0)
        x = dpi_real_to_qmk(-0.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(-1.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(-2.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(-3.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(-4.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(-5.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(-6.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(-7.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        $display("---------------------------------------------------------");
        x = dpi_real_to_qmk(0.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(1.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(2.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(3.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(4.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(5.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(6.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        x = dpi_real_to_qmk(7.2, 4, 4); #10; display_vals(); // -2.5    -> -2
        $display("---------------------------------------------------------");
        

        // 5. Test Maximum Magnitude
        x = 8'sh80; #10; display_vals(); // -8.0    -> -8

        $finish;
    end

    task display_vals;
        begin
            $display("      %d      |   %f  |     %2d      |     %f     |      %d      |       %f      |  %b", 
                     x, dpi_qmk_to_real(x, 4), e_a, e_a/256.0, mantisa, mantisa/16.0, is_zero);
        end
    endtask

endmodule