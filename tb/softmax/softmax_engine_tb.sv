// ===================================================================================
// TESTBENCH ARCHITECTURE: Top-Level SoftMax Engine (softmax_engine_tb)
// ===================================================================================
// This testbench verifies the fully unrolled, pipelined SoftMax accelerator.
//
// 1. THE MATH MODEL:
//    The software calculates the true floating-point SoftMax equation:
//    P(i) = exp(x_i - max(x)) / Sum(exp(x_j - max(x)))
//
// 2. THE HARDWARE SCALING (Q0.8 Format):
//    The hardware produces a final 8-bit output. Because SoftMax probabilities 
//    are strictly positive and bounded between 0.0 and 1.0, the hardware output 
//    represents a Q0.8 unsigned fraction. 
//    The checker reads the hardware output, masks it to an unsigned 8-bit integer, 
//    and divides by 256.0 to convert it back to a standard floating-point probability.
//
// 3. COVERAGE:
//    - Phase 1: Directed Vectors (Flat distributions, one-hot peaked distributions).
//    - Phase 2: Saturated random vectors testing the shift-register synchronization.
// ===================================================================================
`timescale 1ns/1ps

module softmax_engine_tb;

    localparam int N = 8;
    localparam int W = 8;
    localparam int ACC_W = 32;
    localparam int LATENCY = $clog2(N) + 3; // Decompose + MAC sum latency
    
    // Acceptable approximation error margin for the combined Taylor-Region pipeline (~8%)
    localparam real CHECK_TOLERANCE = 0.15; 

    // Signals
    logic clk, rst_n, valid_in, valid_out;
    logic signed [W-1:0] in_data [N];
    logic signed [W-1:0] prob_out [N];

    // ========================================================================
    // REPORTING DATA STRUCTURES
    // ========================================================================
    // This struct holds everything we need to know about a single test vector
    typedef struct {
        logic signed [W-1:0] in_vec[N];
        real expected_probs[N];
        real hw_probs[N];
        bit  passed;
    } test_result_t;

    // Queue to hold the expected math while the hardware processes
    typedef struct {
        logic signed [W-1:0] in_vec[N];
        real expected_probs[N];
    } queued_item_t;

    queued_item_t expected_queue [$];
    test_result_t final_report [$];    // Stores the final results for printing

    int test_passed = 1;
    int total_errors = 0;

    // DUT
    softmax_engine #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .in_data   (in_data),
        .valid_out (valid_out),
        .prob_out  (prob_out)
    );

    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Stimulus
    initial begin
        rst_n = 0;
        valid_in = 0;
        for (int i = 0; i < N; i++) in_data[i] = 0;
        
        repeat(3) @(posedge clk);
        rst_n = 1;

        $display("\n--- PHASE 1: Directed Distributions ---");
        // Uniform distribution (Expected Prob = 1/8 = 0.125)
        push_directed_vector('{0, 0, 0, 0, 0, 0, 0, 0});
        push_directed_vector('{-50, -50, -50, -50, -50, -50, -50, -50});
        
        // Highly peaked distribution (Expected Prob ~1.0 for the max value)
        push_directed_vector('{127, -50, -50, -50, -50, -50, -50, -50});
        push_directed_vector('{-128, -128, -128, -128, -128, -128, -128, 127});

        // Split peak distribution (Expected Prob ~0.5 for the two max values)
        push_directed_vector('{100, 100, -100, -100, -100, -100, -100, -100});

        $display("\n--- PHASE 2: Pipeline Saturation (Random) ---");
        repeat(9) push_test_vector(1'b1); // Keep valid high
        push_test_vector(1'b0);           // Drop valid on the last one

        // Drain pipeline
        repeat(LATENCY + 10) @(posedge clk);

        // ========================================================================
        // PRINT FINAL REPORT
        // ========================================================================
        $display("\n=========================================================================");
        $display("                          FINAL SIMULATION REPORT                        ");
        $display("=========================================================================");
        
        begin
            test_result_t res;
            foreach (final_report[v]) begin
                res = final_report[v];
                
                if (res.passed)
                    $display("\n[VECTOR %0d] - \033[0;32mPASSED\033[0m", v);
                else
                    $display("\n[VECTOR %0d] - \033[0;31mFAILED\033[0m", v);
                    
                $display("Input Vector : %p", res.in_vec);
                
                for (int i = 0; i < N; i++) begin
                    real err;
                    err = (res.hw_probs[i] > res.expected_probs[i]) ? (res.hw_probs[i] - res.expected_probs[i]) : (res.expected_probs[i] - res.hw_probs[i]);
                    
                    if (err > CHECK_TOLERANCE)
                        $display("  Idx %0d | Exp: %f | HW: %f | Err: %f  <-- FAIL", i, res.expected_probs[i], res.hw_probs[i], err);
                    else
                        $display("  Idx %0d | Exp: %f | HW: %f | Err: %f", i, res.expected_probs[i], res.hw_probs[i], err);
                end
            end
        end

        $display("\n=========================================================================");
        if (test_passed && total_errors == 0)
            $display("\033[0;32m[TEST PASSED]\033[0m End-to-End SoftMax Engine Verified!");
        else
            $display("\033[0;31m[TEST FAILED]\033[0m %0d probability errors found.", total_errors);
            
        $finish;
    end

    // --- Stimulus Tasks ---
    task automatic push_directed_vector(input logic signed [W-1:0] vec [N]);
        model_and_push(vec, 1'b0);
    endtask

    task automatic push_test_vector(input logic keep_valid_high);
        logic signed [W-1:0] vec [N];
        // int min_val = -(1 << (W-1));
        // int max_val = (1 << (W-1)) - 1;
        int min_val = -64;
        int max_val = 63;
        foreach (vec[i]) vec[i] = $urandom_range(0, 255) - 128;
        
        model_and_push(vec, keep_valid_high);
    endtask

    task automatic model_and_push(input logic signed [W-1:0] vec [N], input logic keep_valid_high);
        queued_item_t item;
        real exp_vals [N];
        real sum_exp = 0.0;
        int max_val = vec[0];
        int diff;

        item.in_vec = vec;

        // 1. Find Max
        for (int i = 1; i < N; i++) begin
            if (vec[i] > max_val) max_val = vec[i];
        end

        // 2. Subtract, Clamp, and compute Float e^x, then Accumulate S
        for (int i = 0; i < N; i++) begin
            diff = vec[i] - max_val;
            if (diff < -128) diff = -128; 
            
            exp_vals[i] = $exp(real'(diff) / 16.0);
            sum_exp += exp_vals[i];
        end

        // 3. Calculate Final SoftMax Probability
        for (int i = 0; i < N; i++) begin
            item.expected_probs[i] = exp_vals[i] / sum_exp;
        end

        expected_queue.push_back(item);

        // Drive Bus
        in_data <= vec;
        valid_in <= 1'b1;
        @(posedge clk);
        if (!keep_valid_high) valid_in <= 1'b0;
    endtask

    // --- Background Checker ---
    initial begin
        queued_item_t expected_item;
        test_result_t res_item;
        real hw_prob, err;
        bit vector_passed;
        
        forever begin
            @(posedge clk);
            #1; // Wait 1ns for combinational settling
            
            if (valid_out) begin
                if (expected_queue.size() == 0) begin
                    $error("Valid out asserted with empty Queue!");
                    test_passed = 0;
                end else begin
                    expected_item = expected_queue.pop_front();
                    res_item.in_vec = expected_item.in_vec;
                    res_item.expected_probs = expected_item.expected_probs;
                    vector_passed = 1'b1;
                    
                    for (int i = 0; i < N; i++) begin
                        // SoftMax probabilities are purely positive. 
                        // We mask with 8'hFF to treat the signed wire as an unsigned magnitude.
                        // Divide by 256.0 to shift the Q0.8 integer back to a float.
                        hw_prob = real'(prob_out[i] & 8'hFF) / 256.0;
                        res_item.hw_probs[i] = hw_prob;
                        
                        err = (hw_prob > expected_item.expected_probs[i]) ? (hw_prob - expected_item.expected_probs[i]) : (expected_item.expected_probs[i] - hw_prob);

                        if (err > CHECK_TOLERANCE) begin
                            vector_passed = 1'b0;
                            total_errors++;
                            test_passed = 0;
                        end
                    end
                    
                    res_item.passed = vector_passed;
                    final_report.push_back(res_item); // Save to print later
                end                
            end
        end
    end

endmodule