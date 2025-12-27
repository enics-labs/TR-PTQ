`timescale 1ns / 1ps

module ln_approx_q12_4_tb;
    parameter WIDTH = 16;
    parameter BITS  = 4;  // Fractional bits

    // DPI Import
    import "DPI-C" function int c_ln_reference(int xq_val, int K);
    import "DPI-C" function int dpi_real_to_qmk(
        input real real_val,
        input int  M,
        input int  K
    );
    import "DPI-C" function real dpi_qmk_to_real(
        input int fixed_val,
        input int K
    );

    reg  [WIDTH-1:0] xq;
    wire [WIDTH/2-1:0] yq; // This is 8 bits (Q4.4)
    
    int ref_val;
    int error;

    tr_ln #(
      .WIDTH(WIDTH),
      .BITS(BITS)  
    ) uut (
        .xq(xq),
        .yq(yq)
    );

    initial begin
        $display("Testing ln Approximation: Width=16, Bits=4 (Q12.4)");
        $display("Comparing HDL vs. C-DPI Golden Model");
        $display("--------------------------------------------------");

        // Test vectors
        for (int i=10 ; i<1000; i+=10) begin
            test_value(i/10 , WIDTH-BITS, BITS);        
        end
        // test_value(3.4  , WIDTH-BITS, BITS);
        // test_value(5.22 , WIDTH-BITS, BITS);
        // test_value(80.96, WIDTH-BITS, BITS);
        // test_value(120.2, WIDTH-BITS, BITS);

        $finish;
    end

    // Helper task to run tests and compare with DPI
    task test_value(input [WIDTH-1:0] real_val, input int EXP, input int FRAC);
        logic [WIDTH-1:0] fixed_num; 
        real expected_val;
        real actual_val;
        begin
            fixed_num = dpi_real_to_qmk(real_val, EXP, FRAC);
            xq = fixed_num;
            #10;
            ref_val = c_ln_reference(fixed_num, FRAC);
            
            // Note: Since yq is 8-bit, we mask the reference to 8 bits
            // and treat both as signed for error calculation
            error = $signed(yq) - $signed(ref_val[7:0]);
            expected_val = dpi_qmk_to_real(ref_val, FRAC);
            actual_val = dpi_qmk_to_real(yq, FRAC);
            $display("[%f][0x%h] In: 0x%h | HDL: 0x%h | REF: 0x%h | Error: %0d", 
                     real_val, fixed_num, xq, yq, ref_val[7:0], error);
            $display("expected_val [%f], actual_val [%f]", expected_val, actual_val);
            $display("------------------------------------------------------------");
        end
    endtask

endmodule