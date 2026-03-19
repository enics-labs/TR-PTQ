`timescale 1ns/1ps

module tr_norm_engine_tb();

    localparam int N = 8;
    localparam int W = 8;
    localparam int ACC_W = 20;
    localparam int FIFO_DEPTH = 32;

    localparam real CHECK_TOLERANCE = 0.35; 

    // DUT Signals
    logic                 clk, rst_n;
    logic                 valid_in, last_in;
    logic signed [W-1:0]  x_in;
    logic signed [W-1:0]  gamma;
    logic signed [W-1:0]  beta;
    
    logic                 valid_out, last_out;
    logic signed [W-1:0]  y_out;

    tr_norm_engine #(
        .N(N), .W(W), .ACC_W(ACC_W), .FIFO_DEPTH(FIFO_DEPTH)
    ) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    typedef struct {
        logic signed [W-1:0] x_vec[N];
        logic signed [W-1:0] gamma;
        logic signed [W-1:0] beta;
        real                 exp_y[N];
    } queued_item_t;

    queued_item_t expected_queue [$];

    int tests_run;
    int total_errors;
    real max_observed_err;

    initial begin
        tests_run = 0;
        total_errors = 0;
        max_observed_err = 0.0;
        
        rst_n = 0; valid_in = 0; last_in = 0; x_in = 0;
        gamma = 16; beta  = 0; 
        
        #20 rst_n = 1;
        
        $display("=======================================================================");
        $display(" STARTING END-TO-END LAYERNORM ENGINE VERIFICATION (N=%0d)", N);
        $display("=======================================================================");

        $display("\n---> PHASE 1: Directed Vectors (Gamma=1.0, Beta=0.0)...");
        push_vector('{16, 24, 32, 40, 48, 56, 64, 72}, 8'd16, 8'd0); 
        push_vector('{32, 32, 32, 32, 32, 32, 32, 32}, 8'd16, 8'd0); 
        push_vector('{-32, 32, -32, 32, -32, 32, -32, 32}, 8'd16, 8'd0);

        $display("\n---> PHASE 2: Affine Transformation Tests...");
        push_vector('{10, 20, 30, 40, 50, 60, 70, 80}, 8'd8, 8'd32);
        push_vector('{-64, -32, 0, 32, 64, 32, 0, -32}, 8'd32, -8'd16);

        $display("\n---> PHASE 3: Random Pipeline Saturation...");
        begin
            logic signed [W-1:0] rand_gamma;
            logic signed [W-1:0] rand_beta;
            int i; // Hoisted loop iterator
            
            rand_gamma = $urandom_range(8, 32); 
            rand_beta  = $urandom_range(0, 64) - 32;
            
            repeat(500) begin
                logic signed [W-1:0] rand_vec [N];
                for(i=0; i<N; i++) rand_vec[i] = $urandom_range(0, 255) - 128;
                push_vector(rand_vec, rand_gamma, rand_beta);
            end
        end

        $display("\n---> Waiting for pipeline to drain...");
        begin
            int timeout;
            timeout = 0; 
            while (expected_queue.size() > 0 && timeout < 5000) begin
                @(posedge clk);
                timeout++;
            end
            if (expected_queue.size() > 0) $error("FATAL: Pipeline Stalled! Valid_out stuck low.");
        end

        $display("\n=======================================================================");
        $display(" FINAL VERIFICATION REPORT: LAYERNORM ENGINE");
        $display("=======================================================================");
        $display(" Vectors Tested : %0d", tests_run);
        $display(" Max Element Err: %f (Tolerance: %f)", max_observed_err, CHECK_TOLERANCE);
        
        if (total_errors == 0)
            $display("\n \033[0;32m[TEST PASSED]\033[0m End-to-End LayerNorm Engine Verified!");
        else
            $display("\n \033[0;31m[TEST FAILED]\033[0m %0d probability errors found.", total_errors);
            
        $display("=======================================================================\n");
        $finish;
    end

    task automatic push_vector(input logic signed [W-1:0] vec [N], input logic signed [W-1:0] g, input logic signed [W-1:0] b);
        queued_item_t item;
        real sum_x, sum_x_sq;
        real mean, var_val, isqrt, gamma_real, beta_real;
        real x_real, norm_val, final_y; 
        int i; // Hoisted loop iterator
        
        sum_x = 0.0;
        sum_x_sq = 0.0;
        
        item.x_vec = vec;
        item.gamma = g;
        item.beta  = b;
        
        for(i=0; i<N; i++) begin
            real val = real'(vec[i]) / 16.0; 
            sum_x    += val;
            sum_x_sq += (val * val);
        end
        
        mean    = sum_x / N;
        var_val = (sum_x_sq / N) - (mean * mean);
        
        if (var_val <= 0.0) begin
            isqrt = 0.0; 
        end else begin
            isqrt = 1.0 / $sqrt(var_val + (1.0/256.0)); 
        end
        
        gamma_real = real'(g) / 16.0;
        beta_real  = real'(b) / 16.0;
        
        for(i=0; i<N; i++) begin
            x_real = real'(vec[i]) / 16.0;
            norm_val = (x_real - mean) * isqrt;
            
            if (norm_val > 7.9375) norm_val = 7.9375;
            if (norm_val < -8.0)   norm_val = -8.0;
            
            final_y = (norm_val * gamma_real) + beta_real; 
            
            if (final_y > 7.9375) final_y = 7.9375;
            if (final_y < -8.0)   final_y = -8.0;
            
            item.exp_y[i] = final_y;
        end
        
        expected_queue.push_back(item);

        gamma <= g;
        beta  <= b;
        for(i=0; i<N; i++) begin
            valid_in <= 1'b1;
            last_in  <= (i == N-1) ? 1'b1 : 1'b0;
            x_in     <= vec[i];
            @(posedge clk);
        end
        valid_in <= 1'b0;
        last_in  <= 1'b0;
    endtask

    initial begin
        queued_item_t exp;
        real hw_y, err;
        int elem_idx;
        bit vector_failed;
        
        elem_idx = 0; 
        vector_failed = 1'b0;
        
        forever begin
            @(posedge clk); #1; 
            
            if (valid_out) begin
                if (elem_idx == 0) begin
                    if (expected_queue.size() == 0) $error("Valid out asserted with empty Queue!");
                    exp = expected_queue.pop_front();
                    vector_failed = 1'b0;
                end
                
                hw_y = real'(y_out) / 16.0; 
                err = (hw_y > exp.exp_y[elem_idx]) ? (hw_y - exp.exp_y[elem_idx]) : (exp.exp_y[elem_idx] - hw_y);
                
                if (err > max_observed_err) max_observed_err = err;
                
                if (err > CHECK_TOLERANCE) begin
                    $display("[ERROR] Vector Failed! Element %0d | Exp: %6.3f | HW: %6.3f | Err: %6.3f", 
                             elem_idx, exp.exp_y[elem_idx], hw_y, err);
                    vector_failed = 1'b1;
                end
                
                if (last_out) begin
                    if (vector_failed) total_errors++;
                    tests_run++;
                    elem_idx = 0;
                end else begin
                    elem_idx++;
                end
            end
        end
    end

endmodule