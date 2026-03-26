// ===================================================================================
// TESTBENCH ARCHITECTURE: TR-Decomposition & Max Subtraction (online_sum_tb)
// ===================================================================================
// This testbench verifies the dual-mode `online_sum` module.
//
// MODE 1: INTERNAL_MAX = 1 (Pipelined Max Tree)
// MODE 2: INTERNAL_MAX = 0 (Combinational External Max)
//
// Both modes are fed identical vectors. Two parallel background checkers 
// automatically account for the different pipeline latencies and verify that both
// modes perfectly evaluate the Taylor-Region exponential: exp( (x - x_max) / 16.0 )
// ===================================================================================
`timescale 1ns/1ps

module online_sum_tb;

    localparam int DATA_WIDTH = 8;
    localparam int NUM_INPUTS = 8;
    localparam int ITER       = 2;
    localparam int FRAC_W     = 4;
    localparam int LUT_IDX_W  = 3;
    localparam int LATENCY    = $clog2(NUM_INPUTS);

    localparam real CHECK_TOLERANCE = 0.08;

    // Global Signals
    logic clk, rst_n, valid_in;
    logic signed [DATA_WIDTH-1:0] in_data [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] in_data_shifted [NUM_INPUTS]; // For Mode 2
    logic signed [DATA_WIDTH-1:0] ext_x_max;

    // DUT 0 Signals (Mode 0: Int Max, Int Sub)
    logic valid_out_m0;
    logic signed [DATA_WIDTH-1:0] e_a_m0 [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] e_frac_m0 [NUM_INPUTS];

    // DUT 1 Signals (Mode 1: Ext Max, Int Sub)
    logic valid_out_m1;
    logic signed [DATA_WIDTH-1:0] e_a_m1 [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] e_frac_m1 [NUM_INPUTS];

    // DUT 2 Signals (Mode 2: Ext Max, Ext Sub)
    logic valid_out_m2;
    logic signed [DATA_WIDTH-1:0] e_a_m2 [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] e_frac_m2 [NUM_INPUTS];

    typedef real real_array_t [NUM_INPUTS];
    real_array_t expected_queue_m0 [$];
    real_array_t expected_queue_m1 [$];
    real_array_t expected_queue_m2 [$];

    int test_passed;
    int error_count_m0;
    int error_count_m1;
    int error_count_m2;

    // ========================================================================
    // INSTANTIATE ALL 3 MODES
    // ========================================================================
    online_sum #(
        .NUM_INPUTS(NUM_INPUTS), .DATA_WIDTH(DATA_WIDTH),
        .FRAC_W(FRAC_W), .LUT_IDX_W(LUT_IDX_W), .ITER(ITER), .MODE(0)
    ) dut_m0 (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .in_data(in_data), .ext_x_max('0), 
        .valid_out(valid_out_m0), .e_a(e_a_m0), .e_frac(e_frac_m0)
    );

    online_sum #(
        .NUM_INPUTS(NUM_INPUTS), .DATA_WIDTH(DATA_WIDTH),
        .FRAC_W(FRAC_W), .LUT_IDX_W(LUT_IDX_W), .ITER(ITER), .MODE(1)
    ) dut_m1 (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .in_data(in_data), .ext_x_max(ext_x_max), 
        .valid_out(valid_out_m1), .e_a(e_a_m1), .e_frac(e_frac_m1)
    );

    online_sum #(
        .NUM_INPUTS(NUM_INPUTS), .DATA_WIDTH(DATA_WIDTH),
        .FRAC_W(FRAC_W), .LUT_IDX_W(LUT_IDX_W), .ITER(ITER), .MODE(2)
    ) dut_m2 (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .in_data(in_data_shifted), .ext_x_max('0), 
        .valid_out(valid_out_m2), .e_a(e_a_m2), .e_frac(e_frac_m2)
    );

    // ========================================================================
    // STIMULUS & CLOCK
    // ========================================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        int i;
        test_passed = 1;
        error_count_m0 = 0; error_count_m1 = 0; error_count_m2 = 0;
        
        rst_n = 0; valid_in = 0; ext_x_max = 0;
        for (i = 0; i < NUM_INPUTS; i++) begin
            in_data[i] = 0;
            in_data_shifted[i] = 0;
        end
        
        repeat(3) @(posedge clk);
        rst_n = 1;

        $display("=======================================================================");
        $display(" STARTING ONLINE_SUM VERIFICATION (ALL 3 ARCHITECTURAL MODES)");
        $display("=======================================================================");

        $display("\n---> PHASE 1: Directed Edge Cases...");
        push_directed_vector('{0, 0, 0, 0, 0, 0, 0, 0});
        push_directed_vector('{-128, -128, -128, -128, -128, -128, -128, -128});
        push_directed_vector('{127, -50, -50, -50, -50, -50, -50, -50});
        push_directed_vector('{-50, -50, -50, -50, -50, -50, -50, 127});
        push_directed_vector('{-128, 127, -128, 127, -128, 127, -128, 127});

        $display("---> PHASE 2: Pipeline Saturation (Random Back-to-Back)...");
        repeat(9) push_test_vector(1'b1); 
        push_test_vector(1'b0);

        repeat(LATENCY + 5) @(posedge clk);

        $display("\n=======================================================================");
        $display(" FINAL VERIFICATION REPORT: ONLINE SUM (TR-Decomposition)");
        $display("=======================================================================");
        $display(" [MODE 0 : Internal Max + Internal Sub]");
        $display("    -> Errors Found : %0d", error_count_m0);
        $display(" [MODE 1 : External Max + Internal Sub]");
        $display("    -> Errors Found : %0d", error_count_m1);
        $display(" [MODE 2 : External Max + External Sub]");
        $display("    -> Errors Found : %0d", error_count_m2);
        
        if (test_passed && error_count_m0 == 0 && error_count_m1 == 0 && error_count_m2 == 0)
            $display("\n \033[0;32m[TEST PASSED]\033[0m All 3 architectural modes verified successfully!");
        else
            $display("\n \033[0;31m[TEST FAILED]\033[0m Tolerances violated.");
            
        $display("=======================================================================\n");
        $finish;
    end

    task automatic push_directed_vector(input logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS]);
        model_and_push(vec, 1'b0);
    endtask

    task automatic push_test_vector(input logic keep_valid_high);
        logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS];
        int min_val = -(1 << (DATA_WIDTH-1));
        int max_val = (1 << (DATA_WIDTH-1)) - 1;
        int i;
        
        for (i = 0; i < NUM_INPUTS; i++) vec[i] = $urandom_range(max_val, min_val);
        model_and_push(vec, keep_valid_high);
    endtask

    task automatic model_and_push(input logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS], input logic keep_valid_high);
        real_array_t expected_floats;
        logic signed [DATA_WIDTH-1:0] shifted_vec [NUM_INPUTS];
        logic signed [DATA_WIDTH-1:0] max_val;
        int diff, i;

        max_val = vec[0];
        for (i = 1; i < NUM_INPUTS; i++) begin
            if (vec[i] > max_val) max_val = vec[i];
        end

        for (i = 0; i < NUM_INPUTS; i++) begin
            diff = vec[i] - max_val;
            if (diff < -128) diff = -128; 
            
            shifted_vec[i] = diff;
            expected_floats[i] = $exp(real'(diff) / 16.0); 
        end

        expected_queue_m0.push_back(expected_floats);
        expected_queue_m1.push_back(expected_floats);
        expected_queue_m2.push_back(expected_floats);

        in_data         <= vec;
        ext_x_max       <= max_val;
        in_data_shifted <= shifted_vec;
        valid_in        <= 1'b1;
        
        @(posedge clk);
        if (!keep_valid_high) valid_in <= 1'b0;
    endtask

    // ========================================================================
    // BACKGROUND CHECKERS
    // ========================================================================
    initial begin
        real_array_t exp_vals; real hw_val, err; int i;
        forever begin
            @(posedge clk); #1;
            if (valid_out_m0) begin
                if (expected_queue_m0.size() == 0) begin test_passed = 0; end 
                else begin
                    exp_vals = expected_queue_m0.pop_front();
                    for (i = 0; i < NUM_INPUTS; i++) begin
                        hw_val = real'((e_a_m0[i] & 16'hFF) * (e_frac_m0[i] & 16'hFF)) / 4096.0;
                        err = (hw_val > exp_vals[i]) ? (hw_val - exp_vals[i]) : (exp_vals[i] - hw_val);
                        if (err > CHECK_TOLERANCE) begin test_passed = 0; error_count_m0++; end
                    end
                end                
            end
        end
    end

    initial begin
        real_array_t exp_vals; real hw_val, err; int i;
        forever begin
            @(posedge clk); #1;
            if (valid_out_m1) begin
                if (expected_queue_m1.size() == 0) begin test_passed = 0; end 
                else begin
                    exp_vals = expected_queue_m1.pop_front();
                    for (i = 0; i < NUM_INPUTS; i++) begin
                        hw_val = real'((e_a_m1[i] & 16'hFF) * (e_frac_m1[i] & 16'hFF)) / 4096.0;
                        err = (hw_val > exp_vals[i]) ? (hw_val - exp_vals[i]) : (exp_vals[i] - hw_val);
                        if (err > CHECK_TOLERANCE) begin test_passed = 0; error_count_m1++; end
                    end
                end                
            end
        end
    end

    initial begin
        real_array_t exp_vals; real hw_val, err; int i;
        forever begin
            @(posedge clk); #1;
            if (valid_out_m2) begin
                if (expected_queue_m2.size() == 0) begin test_passed = 0; end 
                else begin
                    exp_vals = expected_queue_m2.pop_front();
                    for (i = 0; i < NUM_INPUTS; i++) begin
                        hw_val = real'((e_a_m2[i] & 16'hFF) * (e_frac_m2[i] & 16'hFF)) / 4096.0;
                        err = (hw_val > exp_vals[i]) ? (hw_val - exp_vals[i]) : (exp_vals[i] - hw_val);
                        if (err > CHECK_TOLERANCE) begin test_passed = 0; error_count_m2++; end
                    end
                end                
            end
        end
    end

endmodule