`timescale 1ns/1ps

module transformer_core_top_tb import transformer_ctrl_pkg::*; ();

    localparam int N     = 8;
    localparam int W     = 8;
    localparam int ACC_W = 32;
    localparam int FRAC  = 4;
    
    localparam real SCALE_GELU = 4096.0; // Q4.12
    localparam real SCALE_SMAX = 256.0;  // Q4.4 * Q4.4 = Q8.8
    localparam real SCALE_LN   = 256.0;   // Q4.4

    // ========================================================================
    // CLOCK & RESET
    // ========================================================================
    logic clk;
    logic rst_n;

    always #5 clk = ~clk; // 100MHz Clock

    // ========================================================================
    // DUT SIGNALS
    // ========================================================================
    // Host Interface
    logic       host_req_valid;
    opcode_e    host_opcode;
    logic       host_req_ready;
    logic       host_done_pulse;

    // Data Interface
    logic signed [W-1:0]     a_in [N];
    logic signed [W-1:0]     b_in [N];
    logic                    out_valid;
    logic signed [ACC_W-1:0] out_vec [N];
    logic signed [ACC_W-1:0] out_dot;

    int total_errors = 0;
    int total_tests  = 0;

    // ========================================================================
    // DEVICE UNDER TEST (DUT)
    // ========================================================================
    transformer_core_top #(
        .N(N), .W(W), .ACC_W(ACC_W), .FRAC(FRAC)
    ) u_top (
        .clk             (clk),
        .rst_n           (rst_n),
        .host_req_valid  (host_req_valid),
        .host_opcode     (host_opcode),
        .host_req_ready  (host_req_ready),
        .host_done_pulse (host_done_pulse),
        .a_in            (a_in),
        .b_in            (b_in),
        .out_valid       (out_valid),
        .out_vec         (out_vec),
        .out_dot         (out_dot)
    );

    // ========================================================================
    // FLOATING-POINT GOLDEN MODELS
    // ========================================================================
    // Helper to convert 8-bit Q4.4 integer to real floating point
    function real int_to_real(logic signed [W-1:0] val);
        return real'(val) / 16.0;
    endfunction

    // 1. SoftMax Golden Model
    function void calc_golden_softmax(input logic signed [W-1:0] in_vec[N], output real expected[N]);
        real x_real[N];
        real max_val = -99999.0;
        real exp_sum = 0.0;
        
        for(int i=0; i<N; i++) begin
            x_real[i] = int_to_real(in_vec[i]);
            if (x_real[i] > max_val) max_val = x_real[i];
        end
        for(int i=0; i<N; i++) begin
            exp_sum += $exp(x_real[i] - max_val);
        end
        for(int i=0; i<N; i++) begin
            expected[i] = $exp(x_real[i] - max_val) / exp_sum;
        end
    endfunction

    // 2. LayerNorm Golden Model
    function void calc_golden_layernorm(input logic signed [W-1:0] in_vec[N], output real expected[N]);
        real x_real[N];
        real sum = 0.0;
        real mean, variance = 0.0;
        
        for(int i=0; i<N; i++) begin
            x_real[i] = int_to_real(in_vec[i]);
            sum += x_real[i];
        end
        mean = sum / real'(N);
        
        for(int i=0; i<N; i++) begin
            variance += (x_real[i] - mean) * (x_real[i] - mean);
        end
        variance = variance / real'(N);
        
        for(int i=0; i<N; i++) begin
            // Avoid division by zero
            if (variance == 0.0) expected[i] = 0.0;
            else expected[i] = (x_real[i] - mean) / $sqrt(variance);
        end
    endfunction

    // 3. GELU Golden Model (Using standard approximation: x * sigmoid(1.702 * x))
    function void calc_golden_gelu(input logic signed [W-1:0] in_vec[N], output real expected[N]);
        real x_real;
        for(int i=0; i<N; i++) begin
            x_real = int_to_real(in_vec[i]);
            // Standard GELU approximation or exact erf() can be used here. 
            // Using a high-precision sigmoid approximation for correlation:
            expected[i] = x_real * (1.0 / (1.0 + $exp(-1.702 * x_real)));
        end
    endfunction

    // ========================================================================
    // SOFTWARE DRIVER TASKS
    // ========================================================================
    
    // Hardware Reset
    task reset_system();
        clk = 0;
        rst_n = 0;
        host_req_valid = 0;
        host_opcode = OP_IDLE;
        for (int i = 0; i < N; i++) begin
            a_in[i] = '0;
            b_in[i] = '0;
        end
        #40; 
        rst_n = 1;
        #20;
        $display("[System] Hardware Reset Complete.");
    endtask

    // The Instrcution Driver
    task send_host_command(input opcode_e cmd);
        // 1. Wait for hardware to be ready to accept a command
        wait(host_req_ready == 1'b1);
        @(posedge clk);
        
        // 2. Issue the command
        host_req_valid = 1'b1;
        host_opcode    = cmd;
        @(posedge clk);
        
        // 3. Drop the valid line
        host_req_valid = 1'b0;

        // 4. Block (wait) until the FSM fires the done interrupt
        wait(host_done_pulse == 1'b1);
        @(posedge clk);
    endtask

    // The Master Checker Task
    task run_test_vector(
        input string test_name,
        input logic signed [W-1:0] inputs [N],
        input real expected [N],
        input real tol,
        input int test_mode // 0: GELU, 1: SoftMax, 2: LayerNorm
    );
        $display("\n>>> RUNNING TEST: %s", test_name);
        
        // 1. Load Data
        for (int i = 0; i < N; i++) a_in[i] = inputs[i];

        #40;

        // 2. Execute Hardware Ops
        if (test_mode == 1) begin
            send_host_command(OP_SMAX_P1);
            send_host_command(OP_SMAX_P2);
            send_host_command(OP_SMAX_P3);
        end else if (test_mode == 2) begin
            send_host_command(OP_LN_P1);
            send_host_command(OP_LN_P2);
            send_host_command(OP_LN_P3);
        end else begin
            send_host_command(OP_GELU_P1);
            send_host_command(OP_GELU_P2);
            send_host_command(OP_GELU_P3);
        end

        // 3. Verify Results
        for (int i = 0; i < N; i++) begin
            real actual, diff, scale;

            // Assign dynamic scaling based on the test
            if (test_mode == 1) scale = SCALE_SMAX;
            else if (test_mode == 2) scale = SCALE_LN;
            else scale = SCALE_GELU;

            actual = real'(out_vec[i]) / scale;
            diff = actual - expected[i];
            if (diff < 0) diff = -diff; // Absolute value

            total_tests++;
            if (diff <= tol) begin
                $display("    [PASS] L%0d | In: %6d | Out: %8f | Target: %8f", i, inputs[i], actual, expected[i]);
            end else begin
                $display("    [FAIL] L%0d | In: %6d | Out: %8f | Target: %8f | Diff: %f > %f", 
                         i, inputs[i], actual, expected[i], diff, tol);
                total_errors++;
            end
        end
    endtask

    // ========================================================================
    // VERIFICATION TASK
    // ========================================================================
    real max_quant_error = 0.0; // Track the highest drift seen in the regression

    task run_verification_cycle(
        input string test_name,
        input logic signed [W-1:0] inputs [N],
        input real tol,
        input int test_mode // 0: GELU, 1: SoftMax, 2: LayerNorm
    );
        real expected [N];
        real scale;
        
        $display("\n>>> RUNNING VERIFICATION: %s", test_name);
        
        // 1. Calculate Golden Model
        if (test_mode == 1)      calc_golden_softmax(inputs, expected);
        else if (test_mode == 2) calc_golden_layernorm(inputs, expected);
        else                     calc_golden_gelu(inputs, expected);

        // 2. Load Hardware Data
        for (int i = 0; i < N; i++) a_in[i] = inputs[i];
        #40; // Pipeline delay

        // 3. Execute Hardware Ops
        if (test_mode == 1) begin
            send_host_command(OP_SMAX_P1); send_host_command(OP_SMAX_P2); send_host_command(OP_SMAX_P3);
            scale = SCALE_SMAX;
        end else if (test_mode == 2) begin
            send_host_command(OP_LN_P1); send_host_command(OP_LN_P2); send_host_command(OP_LN_P3);
            scale = SCALE_LN;
        end else begin
            send_host_command(OP_GELU_P1); send_host_command(OP_GELU_P2); send_host_command(OP_GELU_P3);
            scale = SCALE_GELU;
        end

        // 4. Verify & Profile Errors
        for (int i = 0; i < N; i++) begin
            real actual, diff;
            
            actual = real'(out_vec[i]) / scale;
            diff = actual - expected[i];
            if (diff < 0) diff = -diff; // Absolute value

            // Track maximum quantization error
            if (diff > max_quant_error) max_quant_error = diff;

            total_tests++;
            if (diff <= tol) begin
                $display("    [PASS] L%0d | HW: %8f | Golden: %8f | Err: %f", i, actual, expected[i], diff);
            end else begin
                $display("    [FAIL] L%0d | HW: %8f | Golden: %8f | Err: %f > %f", i, actual, expected[i], diff, tol);
                total_errors++;
            end
        end
    endtask

    // // ========================================================================
    // // MAIN EXECUTION THREAD
    // // ========================================================================
    // initial begin
    //     logic signed [W-1:0] test_in [N];
    //     real                 test_exp [N];

    //     $display("======================================================");
    //     $display("  STARTING HARDWARE REGRESSION SUITE");
    //     $display("======================================================");
    //     reset_system();

    //     // --------------------------------------------------------------------
    //     // SWEEP 1: SOFTMAX TESTS
    //     // --------------------------------------------------------------------
        
    //     // 1A. Known Regression (Your proven values)
    //     test_in  = '{8'd0, -8'd16, -8'd32, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128};
    //     test_exp = '{0.6875, 0.2578, 0.0859, 0.0, 0.0, 0.0, 0.0, 0.0};
    //     run_test_vector("SoftMax Regression Profile", test_in, test_exp, 0.05, 1);

    //     // 1B. Uniform Distribution (All inputs equal)
    //     // Expected: 1/8 = 0.125 for all lanes
    //     test_in  = '{8'd16, 8'd16, 8'd16, 8'd16, 8'd16, 8'd16, 8'd16, 8'd16};
    //     test_exp = '{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125};
    //     run_test_vector("SoftMax Uniform Distribution", test_in, test_exp, 0.05, 1);

    //     // 1C. Extreme Saturation (One massive value)
    //     // Expected: Lane 0 approaches 1.0, others approach 0.0
    //     test_in  = '{8'd64, -8'd64, -8'd64, -8'd64, -8'd64, -8'd64, -8'd64, -8'd64};
    //     test_exp = '{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //     run_test_vector("SoftMax Extreme Saturation", test_in, test_exp, 0.05, 1);


    //     // --------------------------------------------------------------------
    //     // SWEEP 2: GELU TESTS
    //     // --------------------------------------------------------------------
        
    //     // 2A. Known Regression (Your proven SU values)
    //     test_in  = '{8'd16, -8'd16, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0};
    //     test_exp = '{0.9179, -0.0820, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //     run_test_vector("GELU Regression Profile", test_in, test_exp, 0.05, 0);

    //     // 2B. Large Positive Bounds (Approaching linear y = x)
    //     // Inputs: 2.0 (32), 3.0 (48). Expected: ~2.0, ~3.0
    //     // We use a wider tolerance (0.15) here because 8-bit quantization drift 
    //     // gets heavier on the upper bounds, but the curve shape should hold.
    //     test_in  = '{8'd32, 8'd48, 8'd64, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0};
    //     test_exp = '{1.95, 2.99, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //     run_test_vector("GELU Positive Upper Bounds", test_in, test_exp, 0.15, 0);

    //     // 2C. Large Negative Bounds (Approaching zero)
    //     // Inputs: -2.0 (-32), -3.0 (-48). Expected: approaches 0
    //     test_in  = '{-8'd32, -8'd48, -8'd64, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0};
    //     test_exp = '{-0.045, -0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //     run_test_vector("GELU Negative Lower Bounds", test_in, test_exp, 0.05, 0);

    //     // --------------------------------------------------------------------
    //     // SWEEP 3: LAYERNORM TESTS
    //     // --------------------------------------------------------------------
        
    //     // 3A. Zero-Mean Symmetric Vector
    //     // Inputs: 4 lanes of 2.0 (32), 4 lanes of -2.0 (-32). 
    //     // Mean = 0. Variance Sum = 8 * (2.0)^2 = 32. 
    //     // ISD calculation: 1 / sqrt(32) ≈ 0.176
    //     // Expected Output: X * 0.176 (approx 0.353 and -0.353)
    //     // *Tolerance is set wide (0.5) so we can read the true hardware output!*
    //     test_in  = '{8'd32, 8'd32, 8'd32, 8'd32, -8'd32, -8'd32, -8'd32, -8'd32};
    //     test_exp = '{1.00, 1.00, 1.00, 1.00, -1.00, -1.00, -1.00, -1.00};
    //     run_test_vector("LayerNorm Zero-Mean Profile", test_in, test_exp, 0.15, 2);

    //     // 3B. Positive Skewed Vector
    //     // Inputs: 4.0, 2.0, 2.0, 0.0 (and zeros). Mean = 1.0 (16).
    //     // The hardware MUST subtract 1.0 from everything before squaring.
    //     // Variance Sum: 3^2 + 1^2 + 1^2 + (-1)^2 + 4*(-1)^2 = 9 + 1 + 1 + 1 + 4 = 16.
    //     // ISD calculation: 1 / sqrt(16) = 0.25
    //     // Expected Outputs: (X - 1.0) * 0.25 -> [0.75, 0.25, 0.25, -0.25, -0.25...]
    //     test_in  = '{8'd64, 8'd32, 8'd32, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0};
    //     test_exp = '{2.12, 0.71, 0.71, -0.71, -0.71, -0.71, -0.71, -0.71};
    //     run_test_vector("LayerNorm Positive Skew", test_in, test_exp, 0.15, 2);

    //     // --------------------------------------------------------------------
    //     // FINAL REPORT
    //     // --------------------------------------------------------------------
    //     $display("\n======================================================");
    //     if (total_errors == 0) begin
    //         $display("  [SUCCESS] REGRESSION PASSED! (%0d/%0d Checkpoints)", total_tests, total_tests);
    //     end else begin
    //         $display("  [ERROR] %0d/%0d FAILURES DETECTED! Check limits.", total_errors, total_tests);
    //     end
    //     $display("======================================================\n");
    //     #100;
    //     $finish;
    // end

    initial begin
        logic signed [W-1:0] test_in [N];

        $display("======================================================");
        $display("  STARTING HARDWARE vs GOLDEN MODEL VERIFICATION");
        $display("======================================================");
        reset_system();

        // --------------------------------------------------------------------
        // DIRECTED EDGE CASES
        // --------------------------------------------------------------------
        test_in = '{8'd32, 8'd32, 8'd32, 8'd32, -8'd32, -8'd32, -8'd32, -8'd32};
        run_verification_cycle("LayerNorm Zero-Mean", test_in, 0.15, 2);

        test_in = '{8'd64, 8'd32, 8'd32, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0};
        run_verification_cycle("LayerNorm Positive Skew", test_in, 0.15, 2);

        // --------------------------------------------------------------------
        // RANDOMIZED MONTE CARLO SWEEP
        // --------------------------------------------------------------------
        $display("\n>>> STARTING RANDOMIZED SWEEP...");
        for (int iter = 0; iter < 10; iter++) begin
            string test_name;
            
            // Generate random Q4.4 inputs (-128 to 127)
            for (int i = 0; i < N; i++) test_in[i] = $urandom_range(0, 127) - 64;
            
            $sformat(test_name, "SoftMax Random Iter %0d", iter);
            run_verification_cycle(test_name, test_in, 0.10, 1);
            
            $sformat(test_name, "LayerNorm Random Iter %0d", iter);
            run_verification_cycle(test_name, test_in, 0.20, 2);
        end

        // --------------------------------------------------------------------
        // FINAL REPORT
        // --------------------------------------------------------------------
        $display("\n======================================================");
        if (total_errors == 0) begin
            $display("  [SUCCESS] DV REGRESSION PASSED! (%0d/%0d Checkpoints)", total_tests, total_tests);
        end else begin
            $display("  [ERROR] %0d/%0d FAILURES DETECTED!", total_errors, total_tests);
        end
        $display("  MAX QUANTIZATION DRIFT SEEN: %f", max_quant_error);
        $display("======================================================\n");
        #100;
        $finish;
    end

endmodule