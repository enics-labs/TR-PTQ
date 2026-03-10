// ===================================================================================
// TESTBENCH ARCHITECTURE: Exponent Decompose + Max Scale (sum_x_max_tb)
// ===================================================================================
// This testbench verifies the `exp_x_minus_xmax` module, which performs the critical
// pre-processing for the SoftMax denominator. 
//
// 1. THE MATH MODEL:
//    The testbench mimics the hardware's 3-step process:
//      A) Find the maximum value in the vector (x_max).
//      B) Subtract x_max from every element (x_shifted = x - x_max).
//      C) Clamp the difference to -128 (-8.0 in Q4.4) to match hardware bounds.
//      D) Compute the golden floating-point exponential: exp(x_shifted / 16.0).
//
// 2. CHECKING STRATEGY (Tolerance):
//    Because the hardware outputs decoupled TR-exp pieces (e_a and e_frac), the 
//    checker must reassemble them: HW_Float = (e_a * e_frac) / 4096.0.
//    It then compares this to the true golden float within a tight CHECK_TOLERANCE.
//
// 3. COVERAGE:
//    Includes Phase 1 directed edge cases (all zeros, extreme bounds) and Phase 2 
//    pipeline saturation (back-to-back vectors) to prove the shift registers correctly 
//    align with the pipelined max tree.
// ===================================================================================
`timescale 1ns/1ps

module online_sum_tb;

    // Parameters
    localparam int DATA_WIDTH = 8;
    localparam int NUM_INPUTS = 8;
    localparam int ITER = 2;
    localparam int LATENCY    = $clog2(NUM_INPUTS);
    // Define color codes as localparams
    localparam string GREEN = "\033[0;32m";
    localparam string RED   = "\033[0;31m";
    localparam string RESET = "\033[0m";
    // TR-exp max error bound
    localparam real CHECK_TOLERANCE = 0.08;

    // Signals
    logic clk, rst_n, valid_in, valid_out;
    logic signed [DATA_WIDTH-1:0] in_data [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] e_a [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] e_frac [NUM_INPUTS];

    // Array of Reals to hold the golden expected floats
    typedef real real_array_t [NUM_INPUTS];
    real_array_t expected_queue [$];

    int test_passed = 1;
    int error_count = 0;

    // Instantiate the Unit Under Test (DUT)
    exp_x_minus_xmax #(
        .NUM_INPUTS(NUM_INPUTS),
        .DATA_WIDTH(DATA_WIDTH),
        .ITER(ITER)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .in_data(in_data),
        .valid_out(valid_out),
        .e_a(e_a),
        .e_frac(e_frac)
    );

    // Clock generation (100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Stimulus process
    initial begin
        init_sim();

        $display("\n--- PHASE 1: Directed Edge Cases ---");
        push_directed_vector('{0, 0, 0, 0, 0, 0, 0, 0});
        push_directed_vector('{-128, -128, -128, -128, -128, -128, -128, -128});
        push_directed_vector('{127, -50, -50, -50, -50, -50, -50, -50});
        push_directed_vector('{-50, -50, -50, -50, -50, -50, -50, 127});
        push_directed_vector('{-128, 127, -128, 127, -128, 127, -128, 127});

        $display("\n--- PHASE 2: Pipeline Saturation (Random) ---");
        repeat(9) push_test_vector(1'b1); // Keep valid_in high between pushes
        push_test_vector(1'b0);

        end_sim();
    end

    task init_sim();
        rst_n = 0;
        valid_in = 0;
        for (int i = 0; i < NUM_INPUTS; i++) in_data[i] = 0;
        
        repeat(3) @(posedge clk);
        rst_n = 1;       
    endtask

    task end_sim();
        repeat(LATENCY + 2) @(posedge clk);

        $display("\n--- Test Summary ---");
        if (test_passed && error_count == 0)
            $display("\033[0;32m[TEST PASSED]\033[0m Decompose and Scale verified.");
        else
            $display("\033[0;31m[TEST FAILED]\033[0m %0d vector errors found.", error_count);
            
        $finish;
    endtask 

    // --- Stimulus Tasks ---
    task automatic push_directed_vector(input logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS]);
        model_and_push(vec, 1'b0);
    endtask

    task automatic push_test_vector(input logic keep_valid_high);
        logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS];
        int min_val = -(1 << (DATA_WIDTH-1));
        int max_val = (1 << (DATA_WIDTH-1)) - 1;
        foreach (vec[i]) vec[i] = $urandom_range(max_val, min_val);
        
        model_and_push(vec, keep_valid_high);
    endtask

    task automatic model_and_push(input logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS], input logic keep_valid_high);
        real_array_t expected_floats;
        int max_val = vec[0];
        int diff;

        // 1. Find max
        for (int i = 1; i < NUM_INPUTS; i++) begin
            if (vec[i] > max_val) max_val = vec[i];
        end

        // 2. Subtract, Clamp, and compute Float
        for (int i = 0; i < NUM_INPUTS; i++) begin
            diff = vec[i] - max_val;
            if (diff < -128) diff = -128; // Hardware Clamp
            expected_floats[i] = $exp(real'(diff) / 16.0);
        end

        expected_queue.push_back(expected_floats);

        in_data <= vec;
        valid_in <= 1'b1;
        @(posedge clk);
        if (!keep_valid_high) valid_in <= 1'b0;
    endtask

    // --- Background Checker ---
    initial begin
        real_array_t exp_vals;
        real hw_val, err;
        
        forever begin
            @(posedge clk);
            #1;

            if (valid_out) begin
                if (expected_queue.size() == 0) begin
                    $error("Valid out asserted with empty Queue!");
                    test_passed = 0;
                end else begin
                    exp_vals = expected_queue.pop_front();
                    
                    for (int i = 0; i < NUM_INPUTS; i++) begin
                        // Recombine e_a and e_frac
                        hw_val = real'((e_a[i] & 16'hFF) * (e_frac[i] & 16'hFF)) / 4096.0;
                        err = (hw_val > exp_vals[i]) ? (hw_val - exp_vals[i]) : (exp_vals[i] - hw_val);

                        if (err > CHECK_TOLERANCE) begin
                            $error("[FAIL] Index %0d | Expected: %f | HW: %f | Err: %f", 
                                   i, exp_vals[i], hw_val, err);
                            test_passed = 0;
                            error_count++;
                        end
                    end
                end                
            end
        end
    end

endmodule
