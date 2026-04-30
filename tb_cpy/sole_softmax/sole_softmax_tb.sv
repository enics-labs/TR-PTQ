`timescale 1ns/1ps

module sole_softmax_tb;

    // ========================================================================
    // PARAMETERS & SIGNALS
    // ========================================================================
    localparam int N = 8;
    localparam int W = 8;
    localparam int FRAC_W = 4;

    logic                    clk;
    logic                    rst_n;
    
    logic                    in_valid;
    logic signed [W-1:0]     in_data [N];
    
    logic                    out_valid;
    logic [W-1:0]            out_data [N];

    // ========================================================================
    // CLOCK GENERATION
    // ========================================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100 MHz (10ns period)
    end

    // ========================================================================
    // DUT INSTANTIATION
    // ========================================================================
    sole_softmax_top #(
        .N(N),
        .W(W),
        .FRAC_W(FRAC_W)
    ) u_dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .in_valid   (in_valid),
        .in_data    (in_data),
        .out_valid  (out_valid),
        .out_data   (out_data)
    );

    // ========================================================================
    // AUTOMATIC HARDWARE MONITOR
    // ========================================================================
    int test_idx = 1;
    always @(posedge clk) begin
        if (out_valid) begin
            real float_val; // 1. Declare the variable cleanly here
            
            $display("\n======================================================");
            $display(">>> RESULTS FOR VECTOR %0d", test_idx);
            $display("======================================================");
            
            for (int i = 0; i < N; i++) begin
                // 2. Assign the dynamic value here
                float_val = real'(out_data[i]) / 16.0; 
                
                $display("  Lane %0d | HW Raw: %3d | Float Prob: %f", 
                         i, out_data[i], float_val);
            end
            test_idx++;
        end
    end

    // ========================================================================
    // TEST SEQUENCE
    // ========================================================================
    initial begin
        // 1. Initialize
        rst_n    = 0;
        in_valid = 0;
        for (int i = 0; i < N; i++) in_data[i] = '0;
        
        // Hold reset
        #20;
        rst_n = 1;
        #10;

        // --------------------------------------------------------------------
        // TEST VECTOR 1: Regression Profile
        // --------------------------------------------------------------------
        $display("\n[TB] Sending Vector 1: Regression Profile...");
        @(posedge clk);
        in_valid = 1;
        in_data  = '{8'sd0, -8'sd16, -8'sd32, -8'sd128, -8'sd128, -8'sd128, -8'sd128, -8'sd128};
        
        @(posedge clk);
        in_valid = 0; // Drop valid, let it flow through the pipeline

        // Wait for pipeline latency (Max Tree + 3 stages = ~6 cycles)
        #100;

        // --------------------------------------------------------------------
        // TEST VECTOR 2: Uniform Distribution
        // --------------------------------------------------------------------
        $display("\n[TB] Sending Vector 2: Uniform Distribution...");
        @(posedge clk);
        in_valid = 1;
        in_data  = '{8'sd16, 8'sd16, 8'sd16, 8'sd16, 8'sd16, 8'sd16, 8'sd16, 8'sd16};
        
        @(posedge clk);
        in_valid = 0;

        // Wait for pipeline to finish
        #100;

        $display("\n[TB] Simulation Complete.");
        $finish;
    end

endmodule