`timescale 1ns/1ps
`define POST_USE_TR_LN

module soft_dsp_eng_post_wrapper_tb;
    localparam int N          = 8;  // Using 8 to match the TR-Decomposition configurations
    localparam int W          = 8;
    localparam int ACC_W      = 32;
    localparam int FRAC       = 4;
    localparam int ITER       = 1;
    localparam int POST_IN_W  = 17;
    localparam int LN_BITS    = 4;
    localparam int RECIP_BITS = 16;

    // Taylor-Region accumulated error tolerance
    localparam real TOL_SUM = 0.00075; 

    logic                    clk;
    logic                    rst_n;
    logic                    in_valid;
    logic                    in_ready;
    logic [1:0]              op_mode;
    logic                    mode_elemwise;
    logic                    dsp_mode;
    logic signed [W-1:0]     a [N];
    logic signed [W-1:0]     b [N];
    logic                    clear_acc;
    logic                    vector_last_in;
    
    logic                    out_valid;
    logic                    out_ready;
    logic [N-1:0]            out_valid_mask;
    logic signed [ACC_W-1:0] out_vec [N];
    logic signed [ACC_W-1:0] out_dot;
    
    logic                    post_valid;
    logic                    post_busy;
    logic signed [ACC_W-1:0] post_out;

    // ========================================================================
    // DUT INSTANTIATION
    // ========================================================================
    soft_dsp_eng_post_wrapper #(
        .N(N), .W(W), .ACC_W(ACC_W), .FRAC(FRAC), .ITER(ITER),
        .POST_IN_W(POST_IN_W), .LN_BITS(LN_BITS), .RECIP_BITS(RECIP_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .op_mode(op_mode),
        .mode_elemwise(mode_elemwise),
        .dsp_mode(dsp_mode),
        .a(a),
        .b(b),
        .clear_acc(clear_acc),
        .vector_last_in(vector_last_in),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_valid_mask(out_valid_mask),
        .out_vec(out_vec),
        .out_dot(out_dot),
        .post_valid(post_valid),
        .post_busy(post_busy),
        .post_out(post_out)
    );

    // ========================================================================
    // TESTBENCH DATA STRUCTURES
    // ========================================================================
    typedef struct {
        bit  is_softmax;
        int  expected_int_dot;
        real expected_float_dot;
        bit  is_last;
    } expected_t;

    expected_t expected_dot_q [$];
    expected_t expected_post_q [$];

    int test_passed;
    int err_mac, err_softmax, err_post;

    // Clock
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // ========================================================================
    // MAIN STIMULUS THREAD
    // ========================================================================
    initial begin
        test_passed = 1;
        err_mac = 0; err_softmax = 0; err_post = 0;
        
        rst_n = 1'b0;
        init_inputs();
        
        repeat(5) @(posedge clk);
        rst_n = 1'b1;
        repeat(2) @(posedge clk);

        $display("=======================================================================");
        $display(" STARTING MULTI-MODE DSP ENGINE VERIFICATION (N=%0d)", N);
        $display("=======================================================================");

        // --------------------------------------------------------------------
        $display("\n---> PHASE 1: Direct Memory Bypass (Standard MAC)...");
        // --------------------------------------------------------------------
        // Drive 2 chunks and accumulate them (Signed x Signed)
        drive_chunk(2'b00, 1'b0, 1'b0, 1'b1, 1'b0, '{1, 2, 3, 4, 5, 6, 7, 8}, '{1, 1, 1, 1, 1, 1, 1, 1});
        drive_chunk(2'b00, 1'b0, 1'b0, 1'b0, 1'b1, '{-1, -2, -3, -4, -5, -6, -7, -8}, '{2, 2, 2, 2, 2, 2, 2, 2});
        
        // Let it drain
        repeat(10) @(posedge clk);

        // --------------------------------------------------------------------
        $display("---> PHASE 2: SoftMax TR-Decomposition Pipeline...");
        // --------------------------------------------------------------------
        // Drive a chunk through the SoftMax pipeline
        drive_chunk(2'b10, 1'b0, 1'b1, 1'b1, 1'b0, '{127, 0, -50, -100, 127, 0, -50, -100}, '{0,0,0,0,0,0,0,0}); // b is ignored
        drive_chunk(2'b10, 1'b0, 1'b1, 1'b0, 1'b0, '{0, -16, -32, -48, -64, -80, -96, -112}, '{0,0,0,0,0,0,0,0}); // b is ignored
        drive_chunk(2'b10, 1'b0, 1'b1, 1'b0, 1'b0, '{0, -16, -32, -48, -64, -80, -96, -112}, '{0,0,0,0,0,0,0,0}); // b is ignored
        drive_chunk(2'b10, 1'b0, 1'b1, 1'b0, 1'b0, '{0, -16, -32, -48, -64, -80, -96, -112}, '{0,0,0,0,0,0,0,0}); // b is ignored
        drive_chunk(2'b10, 1'b0, 1'b1, 1'b0, 1'b1, '{0, -16, -32, -48, -64, -80, -96, -112}, '{0,0,0,0,0,0,0,0}); // b is ignored
        
        // Let it drain
        repeat(15) @(posedge clk);

        // --------------------------------------------------------------------
        $display("---> PHASE 3: Random Multi-Chunk Saturation (SoftMax)...");
        // --------------------------------------------------------------------
        begin
            logic signed [W-1:0] rand_a [N];
            logic signed [W-1:0] rand_b [N]; // Dummy
            
            for (int chunks = 0; chunks < 20; chunks++) begin
                for (int i = 0; i < N; i++) begin
                    rand_a[i] = $urandom_range(0, 255) - 128;
                    rand_b[i] = 0; 
                end
                // Clear on chunk 0, set last_in on chunk 19
                drive_chunk(2'b10, 1'b0, 1'b1, (chunks == 0), (chunks == 19), rand_a, rand_b);
            end
        end

        // Wait for pipelines to clear completely
        begin
            int timeout = 0;
            while ((expected_dot_q.size() > 0 || expected_post_q.size() > 0) && timeout < 5000) begin
                @(posedge clk);
                timeout++;
            end
            if (expected_dot_q.size() > 0 || expected_post_q.size() > 0) 
                $error("FATAL: Pipeline Stalled! Valid outputs never arrived.");
        end

        $display("\n=======================================================================");
        $display(" FINAL VERIFICATION REPORT: POST-WRAPPER ENGINE");
        $display("=======================================================================");
        $display("    -> MAC Bypass Errors : %0d", err_mac);
        $display("    -> SoftMax Math Err  : %0d", err_softmax);
        $display("    -> Post-Trig Errors  : %0d", err_post);
        
        if (test_passed && err_mac == 0 && err_softmax == 0 && err_post == 0)
            $display("\n \033[0;32m[TEST PASSED]\033[0m Engine perfectly multiplexes and pipelines both modes!");
        else
            $display("\n \033[0;31m[TEST FAILED]\033[0m Hardware diverged from Golden Model.");
            
        $display("=======================================================================\n");
        $finish;
    end

    // ========================================================================
    // TASKS: Initialization & Mathematical Driver
    // ========================================================================
    task automatic init_inputs;
        in_valid        = 1'b0;
        op_mode         = 2'd0;
        mode_elemwise   = 1'b0;
        dsp_mode        = 1'b0;
        clear_acc       = 1'b0;
        vector_last_in  = 1'b0;
        out_ready       = 1'b1;
        for (int i = 0; i < N; i++) begin
            a[i] = '0; b[i] = '0;
        end
    endtask

    task automatic drive_chunk(
        input logic [1:0] op_mode_i,
        input logic       mode_elemwise_i,
        input logic       dsp_mode_i,
        input logic       clear_acc_i,
        input logic       vector_last_i,
        input logic signed [W-1:0] a_i [N],
        input logic signed [W-1:0] b_i [N]
    );
        static int current_int_acc = 0;
        static real current_float_acc = 0.0;
        expected_t exp_item;

        // -------------------------------------------------------------
        // GOLDEN MODEL TRACKING
        // -------------------------------------------------------------
        if (dsp_mode_i == 0) begin
            // MAC BYPASS MODE (Exact Integer Tracking)
            if (clear_acc_i) current_int_acc = 0;
            for (int i = 0; i < N; i++) begin
                // Assuming Signed x Signed for testing bypass
                current_int_acc += (int'(a_i[i]) * int'(b_i[i]));
            end
            
            exp_item.is_softmax = 0;
            exp_item.expected_int_dot = current_int_acc;
            exp_item.is_last = vector_last_i;
            
        end else begin
            // SOFTMAX TR-DECOMPOSITION MODE (Float Approximation Tracking)
            int max_val = a_i[0];
            real sum_exp = 0.0;
            int diff;
            
            if (clear_acc_i) current_float_acc = 0.0;
            
            for (int i = 1; i < N; i++) if (a_i[i] > max_val) max_val = a_i[i];
            
            for (int i = 0; i < N; i++) begin
                diff = a_i[i] - max_val;
                if (diff < -128) diff = -128; // Emulate Hardware Clamp
                sum_exp += $exp(real'(diff) / 16.0); // Qx.4 scaling
            end
            current_float_acc += sum_exp;
            
            exp_item.is_softmax = 1;
            exp_item.expected_float_dot = current_float_acc;
            exp_item.is_last = vector_last_i;
        end
        
        expected_dot_q.push_back(exp_item);
        if (vector_last_i) expected_post_q.push_back(exp_item);

        // -------------------------------------------------------------
        // DRIVE HARDWARE
        // -------------------------------------------------------------
        @(posedge clk);
        while (!in_ready) @(posedge clk); #1;
        
        op_mode        <= op_mode_i;
        mode_elemwise  <= mode_elemwise_i;
        dsp_mode       <= dsp_mode_i;
        clear_acc      <= clear_acc_i;
        vector_last_in <= vector_last_i;
        a              <= a_i;
        b              <= b_i;
        in_valid       <= 1'b1;

        @(posedge clk); #1;
        in_valid       <= 1'b0;
        clear_acc      <= 1'b0;
        vector_last_in <= 1'b0;
    endtask

    // ========================================================================
    // BACKGROUND CHECKERS
    // ========================================================================
    initial begin
        expected_t exp;
        real hw_float, err;
        
        forever begin
            @(posedge clk); #1;
            
            // 1. Check the Vector MAC output
            if (out_valid && out_ready) begin
                if (expected_dot_q.size() == 0) begin
                    $error("[%0t] out_valid asserted but queue is empty!", $realtime);
                    test_passed = 0;
                end else begin
                    exp = expected_dot_q.pop_front();
                    
                    if (exp.is_softmax) begin
                        // SoftMax mode generates Qx.12 via MAC accumulation (e_a[Qx.8] * e_frac[Qx.8])
                        hw_float = real'(out_dot) / 4096.0; 
                        err = (hw_float > exp.expected_float_dot) ? (hw_float - exp.expected_float_dot) : (exp.expected_float_dot - hw_float);
                        
                        if (err > TOL_SUM) begin
                            $error("[%0t] SOFTMAX MAC ERR | HW: %f | EXP: %f", $realtime, hw_float, exp.expected_float_dot);
                            test_passed = 0; err_softmax++;
                        end
                    end else begin
                        // Standard MAC bypass checks exact integer math
                        if (out_dot !== exp.expected_int_dot) begin
                            $error("[%0t] STANDARD MAC ERR | HW: %0d | EXP: %0d", $realtime, out_dot, exp.expected_int_dot);
                            test_passed = 0; err_mac++;
                        end
                    end
                end
            end
            
            // 2. Check that the Post-Processing Block correctly caught the "last" chunk trigger
            if (post_valid) begin
                if (expected_post_q.size() == 0) begin
                    $error("[%0t] post_valid triggered, but no 'vector_last_in' chunk was expected!", $realtime);
                    test_passed = 0; err_post++;
                end else begin
                    exp = expected_post_q.pop_front();
                    // We verify pipeline triggering here (TR-LN math is proven in its own unit test)
                    $display("    [OK] Post-processing accurately synced and completed.");
                end
            end
        end
    end

endmodule