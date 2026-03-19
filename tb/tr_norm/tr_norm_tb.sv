`timescale 1ns/1ps

module tr_norm_tb();

    // Parameters matching the DUT
    parameter int N = 5;
    parameter int W = 8;
    parameter int ACC_W = 20;

    // Tolerance thresholds
    localparam real TOL_MEAN = 0.05; // Mean should be highly precise
    localparam real TOL_VAR  = 0.10; // Variance loses some LSBs during squaring/shifting
    localparam real TOL_INV  = 0.15; // Inv Sqrt uses Taylor-Region approximation

    // DUT Signals
    logic                 clk;
    logic                 rst_n;
    logic                 valid_in;
    logic                 last_in;
    logic signed [W-1:0]  x_in;
    
    logic                 valid_out;
    logic signed [W-1:0]  mean_out;
    logic signed [ACC_W-1:0] var_out;
    logic [11:0]          inv_std_dev_out;

    // Instantiate DUT
    tr_norm #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .last_in(last_in),
        .x_in(x_in),
        .valid_out(valid_out),
        .mean_out(mean_out),
        .var_out(var_out),
        .inv_std_dev_out(inv_std_dev_out)
    );

    // Clock Generation (100MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // ========================================================================
    // REPORTING DATA STRUCTURES
    // ========================================================================
    typedef struct {
        logic signed [W-1:0] vec[N];
        real exp_mean;
        real exp_var;
        real exp_isqrt;
    } queued_item_t;

    queued_item_t expected_queue [$];

    int tests_run = 0;
    real max_err_mean = 0.0;
    real max_err_var  = 0.0;
    real max_err_inv  = 0.0;
    int  total_errors = 0;

    // Test Sequence
    initial begin
        rst_n = 0; valid_in = 0; last_in = 0; x_in = 0;
        #20 rst_n = 1;
        
        $display("=======================================================================");
        $display(" STARTING ROBUST TR-NORM VERIFICATION (N=%0d)", N);
        $display("=======================================================================");

        $display("\n---> PHASE 1: Directed Vectors...");
        // 1. Simple Ramp (1.0 to 5.0)
        push_vector('{16, 32, 48, 64, 80}); 
        
        // 2. Flat Vector (Testing Epsilon protection - Var is 0!)
        push_vector('{32, 32, 32, 32, 32}); 
        
        // 3. Zero-Mean Vector
        push_vector('{-32, -16, 0, 16, 32});
        
        // 4. High Variance
        push_vector('{-120, -100, 0, 100, 120});

        $display("---> PHASE 2: Random Saturation...");
        repeat(500) begin
            logic signed [W-1:0] rand_vec [N];
            for(int i=0; i<N; i++) rand_vec[i] = $urandom_range(0, 255) - 128;
            push_vector(rand_vec);
        end

        // Wait for pipeline to drain dynamically
        $display("---> Waiting for pipeline to drain...");
        begin
            int timeout = 0;
            while (expected_queue.size() > 0 && timeout < 5000) begin
                @(posedge clk);
                timeout++;
            end
            if (expected_queue.size() > 0) $error("FATAL: Pipeline Stalled!");
        end

        // Final Report
        $display("\n=======================================================================");
        $display(" VERIFICATION REPORT: TR-NORM (LayerNorm Core)");
        $display("=======================================================================");
        $display(" Vectors Tested : %0d", tests_run);
        $display(" Max Mean Error : %f (Tolerance: %f)", max_err_mean, TOL_MEAN);
        $display(" Max Var Error  : %f (Tolerance: %f)", max_err_var, TOL_VAR);
        $display(" Max ISqrt Error: %f (Tolerance: %f)", max_err_inv, TOL_INV);
        
        if (total_errors == 0)
            $display("\n \033[0;32m[TEST PASSED]\033[0m Streaming LayerNorm verified!");
        else
            $display("\n \033[0;31m[TEST FAILED]\033[0m %0d tolerance violations found.", total_errors);
            
        $display("=======================================================================\n");
        $finish;
    end

    // --- Task to calculate Math Model and push into Hardware ---
    task automatic push_vector(input logic signed [W-1:0] vec [N]);
        queued_item_t item;
        real sum_x = 0.0, sum_x_sq = 0.0;
        
        item.vec = vec;
        
        // Calculate true math model (Working in standard Float)
        for(int i=0; i<N; i++) begin
            real val = real'(vec[i]) / 16.0; // Qx.4
            sum_x    += val;
            sum_x_sq += (val * val);
        end
        
        item.exp_mean = sum_x / N;
        item.exp_var  = (sum_x_sq / N) - (item.exp_mean * item.exp_mean);
        
        // NOTE: Epsilon is irrelevant in hardware implementation,
        //       since variance can be 0.
        //       For the TB to pass we can add the hardware noise gate:
        // if (item.exp_var <= 0.0) begin
        //     item.exp_isqrt = 0.0; 
        // end else begin
        //     // Epsilon is 1 in Qx.8 format (1/256.0)
        //     item.exp_isqrt = 1.0 / $pow(item.exp_var + (1.0/256.0), 0.5);
        // end

        // Epsilon is 1 in Qx.8 format (1/256.0)
        item.exp_isqrt = 1.0 / $pow(item.exp_var + (1.0/256.0), 0.5);
        
        expected_queue.push_back(item);

        // Stream into hardware
        for(int i=0; i<N; i++) begin
            valid_in <= 1'b1;
            last_in  <= (i == N-1) ? 1'b1 : 1'b0;
            x_in     <= vec[i];
            @(posedge clk);
        end
        valid_in <= 1'b0;
        last_in  <= 1'b0;
    endtask

    // --- Background Checker ---
    initial begin
        queued_item_t exp;
        real hw_mean, hw_var, hw_inv;
        real err_mean, err_var, err_inv;
        
        forever begin
            @(posedge clk); #1; 
            
            if (valid_out) begin
                if (expected_queue.size() == 0) begin
                    $error("Valid out asserted with empty Queue!");
                end else begin
                    exp = expected_queue.pop_front();
                    
                    // Convert Hardware outputs to Reals
                    hw_mean = real'(mean_out) / 16.0;        // Qx.4
                    hw_var  = real'(var_out) / 256.0;         // Qx.8
                    hw_inv  = real'(inv_std_dev_out) / 256.0; // Qx.8
                    
                    // Calculate Absolute Errors
                    err_mean = (hw_mean > exp.exp_mean) ? (hw_mean - exp.exp_mean) : (exp.exp_mean - hw_mean);
                    err_var  = (hw_var  > exp.exp_var)  ? (hw_var  - exp.exp_var)  : (exp.exp_var  - hw_var);
                    err_inv  = (hw_inv  > exp.exp_isqrt)? (hw_inv  - exp.exp_isqrt): (exp.exp_isqrt- hw_inv);
                    
                    // Track Max Errors
                    if (err_mean > max_err_mean) max_err_mean = err_mean;
                    if (err_var  > max_err_var)  max_err_var  = err_var;
                    if (err_inv  > max_err_inv)  max_err_inv  = err_inv;
                    
                    // Check Tolerances
                    if (err_mean > TOL_MEAN || err_var > TOL_VAR || err_inv > TOL_INV) begin
                        $display("[ERROR] Vector %p", exp.vec);
                        if (err_mean > TOL_MEAN) $display("   Mean FAIL | Exp: %6.3f, HW: %6.3f", exp.exp_mean, hw_mean);
                        if (err_var  > TOL_VAR)  $display("   Var  FAIL | Exp: %6.3f, HW: %6.3f", exp.exp_var, hw_var);
                        if (err_inv  > TOL_INV)  $display("   InvS FAIL | Exp: %6.3f, HW: %6.3f", exp.exp_isqrt, hw_inv);
                        total_errors++;
                    end
                    
                    tests_run++;
                end
            end
        end
    end

endmodule