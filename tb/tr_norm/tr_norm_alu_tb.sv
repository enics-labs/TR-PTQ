`timescale 1ns/1ps

module tr_norm_alu_tb();

    localparam int N = 8;
    localparam int W = 8;
    localparam int ACC_W = 20;
    localparam int ISQRT_LATENCY = 6;
    localparam real CHECK_TOLERANCE = 0.35; 

    // DUT Signals
    logic                 clk, rst_n;
    logic                 valid_in, last_in, mode;
    logic signed [W-1:0]  x_in;
    
    logic signed [W-1:0]  mean_in;
    logic [11:0]          inv_std_in;
    logic signed [W-1:0]  gamma, beta;
    
    logic                 valid_out, last_out;
    logic signed [W-1:0]  mean_out;
    logic [11:0]          inv_std_dev_out;
    logic signed [W-1:0]  y_out;

    tr_norm_alu #(
        .N(N), .W(W), .ACC_W(ACC_W), .ISQRT_LATENCY(ISQRT_LATENCY)
    ) dut (.*);

    initial clk = 0;
    always #5 clk = ~clk;

    int total_errors = 0;

    initial begin
        rst_n = 0; valid_in = 0; last_in = 0; mode = 0; x_in = 0;
        mean_in = 0; inv_std_in = 0; gamma = 16; beta = 0; 
        
        #20 rst_n = 1;
        
        $display("=======================================================================");
        $display(" STARTING MULTI-PASS LAYERNORM VERIFICATION (N=%0d)", N);
        $display("=======================================================================");

        // Test 1: Simple Directed Vector
        run_multipass_test('{16, 24, 32, 40, 48, 56, 64, 72}, 8'd16, 8'd0);
        
        // Test 2: Flat Vector (0 Variance)
        run_multipass_test('{32, 32, 32, 32, 32, 32, 32, 32}, 8'd16, 8'd0);

        // Test 3: Affine Shift
        run_multipass_test('{-64, -32, 0, 32, 64, 32, 0, -32}, 8'd32, -8'd16);

        // Test 4: Random Stress Test
        $display("---> Running Random Vector Sweep...");
        for (int t = 0; t < 100; t++) begin
            logic signed [W-1:0] rand_vec [N];
            logic signed [W-1:0] rand_g = $urandom_range(8, 32);
            logic signed [W-1:0] rand_b = $urandom_range(0, 64) - 32;
            for(int i=0; i<N; i++) rand_vec[i] = $urandom_range(0, 255) - 128;
            
            run_multipass_test(rand_vec, rand_g, rand_b);
        end

        $display("\n=======================================================================");
        if (total_errors == 0)
            $display(" \033[0;32m[TEST PASSED]\033[0m End-to-End Multi-Pass LayerNorm Verified!");
        else
            $display(" \033[0;31m[TEST FAILED]\033[0m %0d errors found.", total_errors);
        $display("=======================================================================\n");
        $finish;
    end

    task run_multipass_test(input logic signed [W-1:0] vec[N], logic signed [W-1:0] g, logic signed [W-1:0] b);
        real sum_x = 0, sum_x_sq = 0, exp_mean, exp_var, exp_isqrt, norm_val, exp_y[N];
        logic signed [W-1:0] latched_mean;
        logic [11:0] latched_inv_std;
        int i;
        
        // 1. Calculate Expected Math
        for(i=0; i<N; i++) begin
            real val = real'(vec[i]) / 16.0; 
            sum_x += val; sum_x_sq += (val * val);
        end
        exp_mean = sum_x / N;
        exp_var  = (sum_x_sq / N) - (exp_mean * exp_mean);
        exp_isqrt = (exp_var <= 0.0) ? 0.0 : 1.0 / $sqrt(exp_var + (1.0/256.0));
        
        for(i=0; i<N; i++) begin
            norm_val = ((real'(vec[i]) / 16.0) - exp_mean) * exp_isqrt;
            if (norm_val > 7.9375) norm_val = 7.9375;
            if (norm_val < -8.0)   norm_val = -8.0;
            
            exp_y[i] = (norm_val * (real'(g) / 16.0)) + (real'(b) / 16.0);
            if (exp_y[i] > 7.9375) exp_y[i] = 7.9375;
            if (exp_y[i] < -8.0)   exp_y[i] = -8.0;
        end

        // 2. PASS 1 (Mode 0: Statistics)
        mode <= 0;
        for(i=0; i<N; i++) begin
            valid_in <= 1'b1; last_in <= (i == N-1); x_in <= vec[i];
            @(posedge clk);
        end
        valid_in <= 0; last_in <= 0;

        // Wait for stats to pop out
        do begin @(posedge clk); end while (!valid_out);
        latched_mean    = mean_out;
        latched_inv_std = inv_std_dev_out;
        @(posedge clk); // Clear valid

        // 3. PASS 2 (Mode 1: Normalization)
        mode <= 1;
        mean_in <= latched_mean; inv_std_in <= latched_inv_std;
        gamma <= g; beta <= b;

        fork
            begin // Stream Data In
                for(i=0; i<N; i++) begin
                    valid_in <= 1'b1; last_in <= (i == N-1); x_in <= vec[i];
                    @(posedge clk);
                end
                valid_in <= 0; last_in <= 0;
            end
            begin // Catch Data Out & Verify
                for(i=0; i<N; i++) begin
                    do begin @(posedge clk); end while (!valid_out);
                    
                    begin
                        real hw_y = real'(y_out) / 16.0;
                        real err = (hw_y > exp_y[i]) ? (hw_y - exp_y[i]) : (exp_y[i] - hw_y);
                        if (err > CHECK_TOLERANCE) begin
                            $display("[ERROR] Mode 1 Affine Mismatch! Index %0d | Exp: %6.3f | HW: %6.3f", i, exp_y[i], hw_y);
                            total_errors++;
                        end
                    end
                end
            end
        join
    endtask

endmodule