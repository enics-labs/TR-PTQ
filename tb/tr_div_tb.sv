module tb;

    // --------------- DPI ------------------------
    import "DPI-C" function int dpi_reciprocal_ref(
        input int xq,
        input int M,
        input int K
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
    // --------------------------------------------

    localparam int M = 16;
    localparam int K = 4;
    localparam int ITER = 0;

    logic        clk = 0;
    logic        rst_n = 0;

    logic [15:0] xq;
    logic [7:0]  yq_rtl;

    // DUT
    tr_reciprocal #(
        .WIDTH(16),
        .ITER (ITER),
        .BITS (K)
    ) dut (
        .clk (clk),
        .rst_n(rst_n),
        .xq  (xq),
        .yq  (yq_rtl)
    );

    always #5 clk = ~clk;

    logic [M+K-1:0] fixed_num; 

    initial begin
        rst_n = 0;
        #20 rst_n = 1;
        test_value(1, M-K, K);
        test_value(1.1123, M-K, K);
        test_value(2, M-K, K);
        test_value(3, M-K, K);
        test_value(4, M-K, K);
        test_value(5, M-K, K);
        test_value(80, M-K, K);

        $finish;
    end

    task test_value(real x, input int EXP, input int FRAC);        
        int          yq_ref;
        real actual_val;
        real ref_val;

        begin
            // x val input
            xq  = dpi_real_to_qmk(x, EXP, FRAC);

            // Wait for exp pipeline
            repeat (4) @(posedge clk);

            yq_ref     = dpi_reciprocal_ref(xq, 4, FRAC);
            actual_val = dpi_qmk_to_real(yq_rtl, 8);
            ref_val    = dpi_qmk_to_real(yq_ref, 8);

            $display("INUPT (%0h)[%0f]; ref_val (%h)[%0f]; actual_val(%0h)[%0f]",
                xq, x,
                yq_ref, ref_val,
                yq_rtl, actual_val
            );
        end    

    endtask

endmodule
