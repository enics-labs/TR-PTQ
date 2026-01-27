`timescale 1ns/1ps

module tb_vec_mac_requant;

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    localparam int N     = 32;
    localparam int W     = 8;
    localparam int ACC_W = 32;

    // ------------------------------------------------------------
    // DUT signals
    // ------------------------------------------------------------
    logic                     clk;
    logic                     rst_n;

    logic                     in_valid;
    logic                     in_ready;
    logic signed [W-1:0]      a [N];
    logic signed [W-1:0]      b [N];
    logic                     clear_acc;

    logic signed [31:0]       m_0;
    logic signed [4:0]        f_shift;
    logic signed [31:0]       bias;

    logic                     out_valid;
    logic                     out_ready;
    logic signed [7:0]        out_q;

    // ------------------------------------------------------------
    // Instantiate DUT
    // ------------------------------------------------------------
    vec_mac_requant #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (in_valid),
        .in_ready  (in_ready),
        .a         (a),
        .b         (b),
        .clear_acc (clear_acc),
        .m_0       (m_0),
        .f_shift   (f_shift),
        .bias      (bias),
        .out_valid (out_valid),
        .out_ready (out_ready),
        .out_q     (out_q)
    );

    // ------------------------------------------------------------
    // Clock
    // ------------------------------------------------------------
    always #5 clk = ~clk;

    // ------------------------------------------------------------
    // Reference model (matches requant_unit)
    // ------------------------------------------------------------
    function automatic signed [7:0] ref_requant(
        input signed [31:0] acc,
        input signed [31:0] m0,
        input signed [4:0]  f,
        input signed [31:0] bias
    );
        signed [63:0] prod;
        signed [31:0] scaled;
        signed [31:0] shifted;
        signed [31:0] biased;
        begin
            prod    = acc * m0;
            scaled  = (prod + 64'sh40000000) >>> 31; // Q31 rounding
            shifted = scaled >>> f;
            biased  = shifted + bias;

            if (biased > 127)       ref_requant = 127;
            else if (biased < -128) ref_requant = -128;
            else                    ref_requant = biased[7:0];
        end
    endfunction

    // ------------------------------------------------------------
    // Vector dot reference
    // ------------------------------------------------------------
    function automatic signed [31:0] ref_dot(
        input signed [W-1:0] va [N],
        input signed [W-1:0] vb [N]
    );
        signed [63:0] sum;
        begin
            sum = 0;
            for (int i = 0; i < N; i++)
                sum += va[i] * vb[i];
            ref_dot = sum[31:0];
        end
    endfunction

    // ------------------------------------------------------------
    // Drive one vector transaction
    // ------------------------------------------------------------
    task automatic drive_vec(input int seed);
        begin
            for (int i = 0; i < N; i++) begin
                a[i] = $signed($urandom_range(-(1<<(W-2)), (1<<(W-2))-1));
                b[i] = $signed($urandom_range(-(1<<(W-2)), (1<<(W-2))-1));
            end

            clear_acc = 1'b1;
            in_valid  = 1'b1;

            // wait for handshake
            do @(posedge clk); while (!in_ready);

            @(posedge clk);
            in_valid  = 1'b0;
            clear_acc = 1'b0;
        end
    endtask

    // ------------------------------------------------------------
    // Scoreboard
    // ------------------------------------------------------------
    signed [31:0] exp_acc;
    signed [7:0]  exp_q;

    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            if (out_q !== exp_q) begin
                $error("❌ MISMATCH: expected=%0d got=%0d", exp_q, out_q);
                $fatal;
            end else begin
                $display("✅ PASS: out_q=%0d", out_q);
            end
        end
    end

    // ------------------------------------------------------------
    // Test sequence
    // ------------------------------------------------------------
    initial begin
        clk       = 0;
        rst_n     = 0;
        in_valid  = 0;
        clear_acc = 0;
        out_ready = 1;

        // Fixed quant params (Q31 multiplier)
        m_0     = 32'sd1073741824; // 0.5 in Q31
        f_shift = 3;
        bias    = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;

        // --------------------------------------------------------
        // Run multiple vectors
        // --------------------------------------------------------
        repeat (20) begin
            drive_vec($urandom);

            // compute expected
            exp_acc = ref_dot(a, b);
            exp_q   = ref_requant(exp_acc, m_0, f_shift, bias);

            // random backpressure
            repeat ($urandom_range(0,3)) begin
                out_ready = 0;
                @(posedge clk);
            end
            out_ready = 1;

            // wait for output
            wait (out_valid);
            @(posedge clk);
        end

        $display("🎉 ALL TESTS PASSED");
        $finish;
    end

endmodule
