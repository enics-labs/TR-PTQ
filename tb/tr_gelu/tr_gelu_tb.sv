`timescale 1ns/1ps

module tr_gelu_tb;

    localparam int W = 8;
    localparam int FRAC_W = 4;
    localparam real CHECK_TOLERANCE = 0.15; // Adjusted for aggressive 4-bit exp approximation

    // Signals
    logic clk, rst_n, valid_in, valid_out;
    logic signed [W-1:0] x_in;
    logic signed [W-1:0] gelu_out;

    // ========================================================================
    // REPORTING DATA STRUCTURES
    // ========================================================================
    typedef struct {
        logic signed [W-1:0] x_in;
        real expected_val;
        real hw_val;
        bit  passed;
    } test_result_t;

    typedef struct {
        logic signed [W-1:0] x_in;
        real expected_val;
    } queued_item_t;

    queued_item_t expected_queue [$];
    test_result_t final_report [$];

    int test_passed = 1;
    int total_errors = 0;

    // DUT
    tr_gelu #(
        .W(W)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .x_in      (x_in),
        .valid_out (valid_out),
        .gelu_out  (gelu_out)
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
        x_in = 0;
        
        repeat(3) @(posedge clk);
        rst_n = 1;

        // PHASE 1: Directed Values (Edge cases and typical ranges)
        // Values in Q4.4 format (16 = 1.0, -16 = -1.0)
        model_and_push(8'd0);    //  0.0
        model_and_push(8'd16);   //  1.0
        model_and_push(-8'd16);  // -1.0
        model_and_push(8'd32);   //  2.0
        model_and_push(-8'd32);  // -2.0
        model_and_push(8'd64);   //  4.0
        model_and_push(-8'd64);  // -4.0
        model_and_push(8'h7F);   // Max positive (~7.93)
        model_and_push(8'h80);   // Max negative (-8.0)

        // PHASE 2: Random Saturation (Safe signed generation)
        repeat(20) begin
            model_and_push($urandom_range(0, 255) - 128);
        end

        // Drain pipeline (Wait enough cycles for latency)
        repeat(20) @(posedge clk);

        // ========================================================================
        // PRINT FINAL REPORT
        // ========================================================================
        $display("\n=========================================================================");
        $display("                          TR-GELU SIMULATION REPORT                      ");
        $display("=========================================================================");
        
        begin
            test_result_t res;
            real err; // <-- MOVED UP HERE to satisfy Xcelium scope rules
            
            foreach (final_report[v]) begin
                res = final_report[v];
                
                err = (res.hw_val > res.expected_val) ? (res.hw_val - res.expected_val) : (res.expected_val - res.hw_val);
                
                if (res.passed)
                    $display("[TEST %0d] \033[0;32mPASSED\033[0m | x_in: %4d | Exp: %6.3f | HW: %6.3f | Err: %6.3f", 
                             v, res.x_in, res.expected_val, res.hw_val, err);
                else
                    $display("[TEST %0d] \033[0;31mFAILED\033[0m | x_in: %4d | Exp: %6.3f | HW: %6.3f | Err: %6.3f", 
                             v, res.x_in, res.expected_val, res.hw_val, err);
            end
        end

        $display("\n=========================================================================");
        if (test_passed && total_errors == 0)
            $display("\033[0;32m[TEST PASSED]\033[0m End-to-End TR-GELU Engine Verified!");
        else
            $display("\033[0;31m[TEST FAILED]\033[0m %0d errors found.", total_errors);
            
        $finish;
    end

    // --- Stimulus Tasks ---
    task automatic model_and_push(input logic signed [W-1:0] val);
        queued_item_t item;
        real x_real;
        real expected_gelu;

        item.x_in = val;
        x_real = real'(val) / 16.0; // Convert Q4.4 to Float

        // GELU Approximation: x * sigmoid(1.702 * x)
        // Standard sigmoid: 1 / (1 + exp(-x))
        expected_gelu = x_real / (1.0 + $exp(-1.702 * x_real));

        item.expected_val = expected_gelu;
        expected_queue.push_back(item);

        x_in <= val;
        valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;
    endtask

    // --- Background Silent Checker ---
    initial begin
        queued_item_t expected_item;
        test_result_t res_item;
        real hw_gelu, err;
        
        forever begin
            @(posedge clk);
            #1; 
            
            if (valid_out) begin
                if (expected_queue.size() == 0) begin
                    $error("Valid out asserted with empty Queue!");
                    test_passed = 0;
                end else begin
                    expected_item = expected_queue.pop_front();
                    res_item.x_in = expected_item.x_in;
                    res_item.expected_val = expected_item.expected_val;
                    res_item.passed = 1'b1;
                    
                    // Convert hardware output from Q4.4 back to float
                    hw_gelu = real'(gelu_out) / 16.0;
                    res_item.hw_val = hw_gelu;
                    
                    err = (hw_gelu > expected_item.expected_val) ? (hw_gelu - expected_item.expected_val) : (expected_item.expected_val - hw_gelu);

                    if (err > CHECK_TOLERANCE) begin
                        res_item.passed = 1'b0;
                        total_errors++;
                        test_passed = 0;
                    end
                    
                    final_report.push_back(res_item);
                end                
            end
        end
    end

endmodule