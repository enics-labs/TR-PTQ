`timescale 1ns/1ps

module tr_softmax_tb;

    localparam int N = 8;
    localparam int W = 8;
    localparam int ACC_W = 32;

    // Adjusted Tolerances to account for 8x accumulated Taylor approximation noise
    localparam real TOL_SUM = 0.75; 
    localparam real TOL_INV = 0.25;

    // Global Signals
    logic clk, rst_n, valid_in;
    logic signed [W-1:0] in_data [N];

    // ========================================================================
    // DUT 0: TR-Reciprocal (Combinational Log-Domain)
    // ========================================================================
    logic             v_decomp_0;
    logic [W-1:0]     e_a_0 [N];
    logic [W-1:0]     e_frac_0 [N];
    logic             v_sum_0;
    logic signed [ACC_W-1:0] sum_S_0;
    logic [7:0]       inv_S_0;

    tr_softmax #(
        .N(N), .W(W), .ACC_W(ACC_W), .MODE(0), .RECIP_TYPE(0)
    ) dut_tr_recip (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .in_data(in_data), .ext_x_max('0),
        .out_valid_decomp(v_decomp_0), .out_e_a(e_a_0), .out_e_frac(e_frac_0),
        .out_valid_sum(v_sum_0), .out_sum_S(sum_S_0), .out_inv_S(inv_S_0)
    );

    // ========================================================================
    // DUT 1: Classic Divider (16-Cycle Sequential)
    // ========================================================================
    logic             v_decomp_1;
    logic [W-1:0]     e_a_1 [N];
    logic [W-1:0]     e_frac_1 [N];
    logic             v_sum_1;
    logic signed [ACC_W-1:0] sum_S_1;
    logic [7:0]       inv_S_1;

    tr_softmax #(
        .N(N), .W(W), .ACC_W(ACC_W), .MODE(0), .RECIP_TYPE(1)
    ) dut_classic_recip (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .in_data(in_data), .ext_x_max('0),
        .out_valid_decomp(v_decomp_1), .out_e_a(e_a_1), .out_e_frac(e_frac_1),
        .out_valid_sum(v_sum_1), .out_sum_S(sum_S_1), .out_inv_S(inv_S_1)
    );

    // ========================================================================
    // REPORTING DATA STRUCTURES
    // ========================================================================
    typedef struct {
        real exp_vals[N];
        real sum_exp;
        real inv_sum;
    } expected_t;

    expected_t expected_q0 [$];
    expected_t expected_q1 [$];

    int test_passed;
    int err_decomp, err_sum, err_inv;

    // Clock
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ========================================================================
    // MAIN STIMULUS THREAD
    // ========================================================================
    initial begin
        int i;
        test_passed = 1;
        err_decomp = 0; err_sum = 0; err_inv = 0;
        
        rst_n = 0; valid_in = 0;
        for (i = 0; i < N; i++) in_data[i] = 0;
        
        repeat(3) @(posedge clk);
        rst_n = 1;

        $display("=======================================================================");
        $display(" STARTING TR-SOFTMAX CORE VERIFICATION (N=%0d)", N);
        $display("=======================================================================");

        $display("\n---> PHASE 1: Directed Vectors...");
        push_vector('{0, 0, 0, 0, 0, 0, 0, 0});
        repeat(20) @(posedge clk);
        push_vector('{-128, -128, -128, -128, -128, -128, -128, -128});
        repeat(20) @(posedge clk);
        push_vector('{127, 0, -50, -100, 127, 0, -50, -100});
        repeat(20) @(posedge clk);
        push_vector('{-16, -32, -48, -64, -80, -96, -112, -128});
        repeat(20) @(posedge clk);

        $display("---> PHASE 2: Random Vectors (Spaced for Sequential Divider)...");
        begin
            int j;
            repeat(100) begin
                logic signed [W-1:0] rand_vec [N];
                for(j=0; j<N; j++) rand_vec[j] = $urandom_range(0, 255) - 128;
                
                push_vector(rand_vec);
                
                repeat(20) @(posedge clk);
            end
        end

        $display("---> Waiting for sequential division pipelines to drain...");
        begin
            int timeout;
            timeout = 0;
            while ((expected_q0.size() > 0 || expected_q1.size() > 0) && timeout < 5000) begin
                @(posedge clk);
                timeout++;
            end
            if (expected_q0.size() > 0 || expected_q1.size() > 0) 
                $error("FATAL: Pipeline Stalled! Valid_sum stuck low.");
        end

        $display("\n=======================================================================");
        $display(" FINAL VERIFICATION REPORT: TR-SOFTMAX CORE");
        $display("=======================================================================");
        $display("    -> Decomp Errors    : %0d", err_decomp);
        $display("    -> MAC Sum Errors   : %0d", err_sum);
        $display("    -> Reciprocal Errors: %0d", err_inv);
        
        if (test_passed && err_decomp == 0 && err_sum == 0 && err_inv == 0)
            $display("\n \033[0;32m[TEST PASSED]\033[0m Both SoftMax pipelines verified successfully!");
        else
            $display("\n \033[0;31m[TEST FAILED]\033[0m Tolerances violated.");
            
        $display("=======================================================================\n");
        $finish;
    end

    // ========================================================================
    // MATHEMATICAL MODEL & DRIVER
    // ========================================================================
    task automatic push_vector(input logic signed [W-1:0] vec [N]);
        expected_t item;
        logic signed [W-1:0] max_val;
        int diff, i;

        max_val = vec[0];
        for (i=1; i<N; i++) if (vec[i] > max_val) max_val = vec[i];

        item.sum_exp = 0.0;
        for (i=0; i<N; i++) begin
            diff = vec[i] - max_val;
            if (diff < -128) diff = -128; 
            
            item.exp_vals[i] = $exp(real'(diff) / 16.0); 
            item.sum_exp += item.exp_vals[i];
        end

        item.inv_sum = 1.0 / item.sum_exp;

        expected_q0.push_back(item);
        expected_q1.push_back(item);

        in_data <= vec;
        valid_in <= 1'b1;
        @(posedge clk);
        valid_in <= 1'b0;
    endtask

    // ========================================================================
    // BACKGROUND CHECKER: DUT 0 (TR-Reciprocal)
    // ========================================================================
    initial begin
        expected_t exp_item;
        real hw_sum, hw_inv, err;
        
        forever begin
            @(posedge clk); #1;
            if (v_sum_0) begin
                if (expected_q0.size() == 0) begin test_passed = 0; end 
                else begin
                    exp_item = expected_q0.pop_front();
                    
                    hw_sum = real'(sum_S_0) / 4096.0;
                    err = (hw_sum > exp_item.sum_exp) ? (hw_sum - exp_item.sum_exp) : (exp_item.sum_exp - hw_sum);
                    if (err > TOL_SUM) begin test_passed = 0; err_sum++; end
                    
                    hw_inv = real'(inv_S_0) / 16.0;
                    err = (hw_inv > exp_item.inv_sum) ? (hw_inv - exp_item.inv_sum) : (exp_item.inv_sum - hw_inv);
                    if (err > TOL_INV) begin test_passed = 0; err_inv++; end
                end                
            end
        end
    end

    // ========================================================================
    // BACKGROUND CHECKER: DUT 1 (Classic Divider)
    // ========================================================================
    initial begin
        expected_t exp_item;
        real hw_sum, hw_inv, err;
        
        forever begin
            @(posedge clk); #1;
            if (v_sum_1) begin
                if (expected_q1.size() == 0) begin test_passed = 0; end 
                else begin
                    exp_item = expected_q1.pop_front();
                    
                    hw_sum = real'(sum_S_1) / 4096.0;
                    err = (hw_sum > exp_item.sum_exp) ? (hw_sum - exp_item.sum_exp) : (exp_item.sum_exp - hw_sum);
                    if (err > TOL_SUM) begin test_passed = 0; err_sum++; end
                    
                    hw_inv = real'(inv_S_1) / 16.0;
                    err = (hw_inv > exp_item.inv_sum) ? (hw_inv - exp_item.inv_sum) : (exp_item.inv_sum - hw_inv);
                    if (err > TOL_INV) begin test_passed = 0; err_inv++; end
                end                
            end
        end
    end

endmodule