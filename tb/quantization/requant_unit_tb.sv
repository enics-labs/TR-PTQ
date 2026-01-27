module tb_requant_unit;

    // Clock and Reset
    logic clk;
    logic rst_n;

    // DUT signals
    logic signed [31:0] acc_sum;
    logic signed [31:0] m_0;
    logic signed [4:0]  f_shift;
    logic signed [31:0] bias;
    
    logic in_valid, in_ready;
    logic out_valid, out_ready;
    logic signed [7:0] out_quant;

    // ------------------------------------------------------------
    // Clock Generation
    // ------------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz

    // ------------------------------------------------------------
    // Instantiate DUT (3-Stage Pipeline)
    // ------------------------------------------------------------
    requant_unit dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (in_valid),
        .in_ready (in_ready),
        .out_ready(out_ready),
        .out_valid(out_valid),
        .acc_sum  (acc_sum),
        .m_0      (m_0),
        .f_shift  (f_shift),
        .bias     (bias),
        .out_quant(out_quant)
    );

    // Reference model function (Same as before)
    function automatic signed [7:0] ref_requant(
        input signed [31:0] acc, input signed [31:0] mul,
        input signed [4:0]  sh,  input signed [31:0] b
    );
        logic signed [63:0] prod    = acc * mul;
        logic signed [63:0] scaled  = (prod + 64'sh40000000) >>> 31;
        logic signed [63:0] shifted = scaled >>> sh;
        logic signed [63:0] biased  = shifted + b;
        if (biased > 127) return 8'sd127;
        else if (biased < -128) return -8'sd128;
        else return biased[7:0];
    endfunction

    // ------------------------------------------------------------
    // Test Task (Pipelined)
    // ------------------------------------------------------------
    task automatic run_test(
        input signed [31:0] acc,
        input signed [31:0] mul,
        input signed [4:0]  sh,
        input signed [31:0] b
    );
        signed [7:0] exp;
        
        // 1. Wait for DUT to be ready
        while (!in_ready) @(posedge clk);

        // 2. Drive Inputs
        acc_sum  = acc;
        m_0      = mul;
        f_shift  = sh;
        bias     = b;
        in_valid = 1;

        @(posedge clk);
        in_valid = 0;

        // 3. Wait for Valid Output (Latency)
        while (!out_valid) @(posedge clk);

        // 4. Check Result
        exp = ref_requant(acc, mul, sh, b);
        if (out_quant !== exp) begin
            $error("FAIL: acc=%0d | exp=%0d got=%0d", acc, exp, out_quant);
        end else begin
            $display("PASS: acc=%0d | out=%0d", acc, out_quant);
        end
    endtask

    // ------------------------------------------------------------
    // Test Sequence
    // ------------------------------------------------------------
    initial begin
        // Init signals
        rst_n = 0;
        in_valid = 0;
        out_ready = 1; // Always ready to receive
        
        repeat(5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        $display("==== Starting Pipelined Tests ====");

        run_test(32'sd1000,  32'sh40000000, 0, 0);   
        run_test(32'sd1000,  32'sh40000000, 0, 10);
        run_test(32'sd1000,  32'sh40000000, 3, 0);

        repeat(100) begin
            run_test($urandom(), $urandom(), $urandom_range(0,31), $urandom());
        end

        #100;
        $display("==== ALL TESTS DONE ====");
        $finish;
    end

endmodule