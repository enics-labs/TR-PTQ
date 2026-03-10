// ===================================================================================
// TESTBENCH ARCHITECTURE: Dual-Mode Exponential Sum (exp_sum_tb)
// ===================================================================================
// This testbench verifies the `exp_sum` module, which acts as a multiplexed bridge 
// between the Taylor-Region Exponential (TR-exp) blocks and the Vector MAC engine.
// It must verify two completely different mathematical domains routed through the 
// same hardware pipeline.
//
// 1. DUAL-MODE COVERAGE & GENERATION:
//    * Mode 0 (MAC Bypass): Routes `x1` and `x2` directly to the MAC. The TB generates
//      full-range random signed integers to test standard dot-product functionality.
//    * Mode 1 (EXP_SUM): Routes TR-exp outputs (e_a and mantisa) to the MAC to 
//      calculate the SoftMax denominator. The TB generates strictly negative Q4.4 
//      values spanning the full supported hardware range (0.0 down to -8.0).
//
// 2. THE GOLDEN MATH MODELS & CHECKING STRATEGY:
//    Because the pipeline processes data over multiple clock cycles (LATENCY = 3), 
//    the `op_queue` tracks which mode was pushed so the checker applies the correct math:
//    
//    * Strict Integer Check (MAC Mode): Calculates the exact signed dot product in 
//      software. The checker demands a 100% bit-accurate match (===).
//
//    * Float Tolerance Check (EXP Mode): The TR-exp hardware is a piecewise polynomial 
//      approximation, so it will never perfectly match a true math $exp() function.
//      Instead, the TB calculates the true floating-point sum of e^x. 
//      Crucially, the hardware defers the final >> 4 bit-shift to prevent precision loss.
//      Therefore, the hardware output is scaled by 2^12 (2^8 from the LUT * 2^4 from Q4.4).
//      The checker divides the hardware output by 4096.0 to convert it back to a true 
//      float, and verifies it falls within a defined CHECK_TOLERANCE.
//
// 3. TYPING NOTE:
//    SystemVerilog strict-typing requires the `vec1` and `vec2` arrays to be explicitly 
//    declared as `signed` to successfully connect to the DUT's `signed [W-1:0] x1 [N]` ports.
// ===================================================================================
module exp_sum_tb;

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    localparam int DATA_WIDTH = 8;
    localparam int NUM_INPUTS = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int NUM_TV     = 10;
    localparam int LATENCY    = 3; // vec_mac_dsp48 pipeline depth
    parameter      FRAC       = 4;
    parameter      ITER       = 2;

    // Max error per item is ~0.07. 8 items per vector = ~0.56 max error per cycle.
    // Over a 10-cycle accumulation window, the total accumulated error could reach ~5.6.
    localparam real CHECK_TOLERANCE = 5.6;

    // ------------------------------------------------------------
    // Signals
    // ------------------------------------------------------------
    logic clk;
    logic rst_n;

    logic valid_in;
    logic in_ready;
    logic clear_acc;
    logic op_type;

    logic out_valid;
    logic out_ready;

    logic signed [DATA_WIDTH-1:0] vec1 [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] vec2 [NUM_INPUTS];
    logic signed [ACC_WIDTH-1:0]  out_data;

    // ------------------------------------------------------------
    // Tracking Queues for the Pipeline
    // ------------------------------------------------------------
    logic                        op_queue       [$]; // Tracks if the output is EXP or MAC
    logic signed [ACC_WIDTH-1:0] expected_mac_q [$]; // Golden integers for MAC mode
    real                         expected_exp_q [$]; // Golden floats for EXP mode

    real                         expected_float_acc;
    logic signed [ACC_WIDTH-1:0] expected_int_acc;
    int                          test_passed;

    // ------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------

    exp_sum #(
        .FRAC(FRAC),
        .ITER(ITER),
        .N(NUM_INPUTS),
        .W(DATA_WIDTH),
        .ACC_W(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(valid_in),
        .in_ready(in_ready),
        .clear_acc(clear_acc),
        .op_type(op_type),
        .x1(vec1),
        .x2(vec2),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_dot(out_data)
    );

    // ------------------------------------------------------------
    // Clock & Initialization
    // ------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        test_passed = 1;
        out_ready   = 1'b1;
        valid_in    = 0;
        clear_acc   = 0;
        op_type     = 0;
        rst_n       = 0;
        expected_float_acc = 0.0;
        expected_int_acc   = 0;

        // Reset
        repeat (3) @(posedge clk);
        rst_n = 1;

        // --------------------------------------------------------
        // TEST 1: MAC Bypass Mode (op_type = 0)
        // --------------------------------------------------------
        $display("\n--- Testing MAC Bypass Mode (Exact Integer Check) ---");
        for (int t = 0; t < NUM_TV; t++) 
            push_vector(t == 0, 1'b0);
        
        // Wait for pipeline to drain before switching modes
        repeat (LATENCY + 2) @(posedge clk);

        // --------------------------------------------------------
        // TEST 2: Exponential Sum Mode (op_type = 1)
        // --------------------------------------------------------
        $display("--- Testing EXP_SUM Mode (Float Tolerance Check) ---");
        for (int t = 0; t < NUM_TV; t++) 
            push_vector(t == 0, 1'b1);

        // Drain pipeline
        repeat (LATENCY + 2) @(posedge clk);

        if (test_passed)
            $display("\n\033[0;32m[TEST PASSED]\033[0m");
        else
            $display("\n\033[0;31m[TEST FAILED]\033[0m");

        $finish;
    end

    // ------------------------------------------------------------
    // Push Vector & Math Model
    // ------------------------------------------------------------
    task automatic push_vector(input bit first, input logic mode);
        logic signed [DATA_WIDTH-1:0] v1 [NUM_INPUTS];
        logic signed [DATA_WIDTH-1:0] v2 [NUM_INPUTS];
        
        logic signed [ACC_WIDTH-1:0] int_dot = 0;
        real float_dot = 0.0;
        real real_x;

        generate_random_vector(v1, mode);
        generate_random_vector(v2, mode);

        if (mode == 1'b0) begin 
            // MAC MODE MATH (Signed Integer Dot Product)
            for (int i = 0; i < NUM_INPUTS; i++) begin
                int_dot += $signed(v1[i]) * $signed(v2[i]);
            end
            
            if (first) expected_int_acc = int_dot;
            else       expected_int_acc += int_dot;
            
            expected_mac_q.push_back(expected_int_acc);
        end else begin          
            // EXP MODE MATH (Float Accumulation)
            for (int i = 0; i < NUM_INPUTS; i++) begin
                // Convert Q4.4 directly to real: divide signed integer by 16.0
                real_x = real'($signed(v1[i])) / 16.0;
                float_dot += $exp(real_x);
            end
            
            if (first) expected_float_acc = float_dot;
            else       expected_float_acc += float_dot;
            
            expected_exp_q.push_back(expected_float_acc);
        end

        // Track the operation type for the checker
        op_queue.push_back(mode);

        // Drive the bus
        @(posedge clk);
        valid_in  <= 1'b1;
        clear_acc <= first;
        op_type   <= mode;
        vec1      <= v1;
        vec2      <= v2;

        @(posedge clk);
        valid_in  <= 0;
        clear_acc <= 0;
    endtask

    // ------------------------------------------------------------
    // Background Checker
    // ------------------------------------------------------------
    initial begin
        logic mode;
        logic signed [ACC_WIDTH-1:0] exp_i;
        real exp_f, hw_f, error;

        forever begin
            @(posedge clk);
            if (out_valid) begin
                if (op_queue.size() == 0) begin
                    $error("Unexpected output received!");
                    test_passed = 0;
                end else begin
                    mode = op_queue.pop_front();

                    if (mode == 1'b0) begin
                        // STRICT INTEGER CHECK FOR MAC
                        exp_i = expected_mac_q.pop_front();
                        if (out_data !== exp_i) begin
                            $error("[MAC FAIL] Expected=%0d, Got=%0d", exp_i, out_data);
                            test_passed = 0;
                        end
                    end else begin
                        // TOLERANCE CHECK FOR EXP
                        exp_f = expected_exp_q.pop_front();
                        
                        // Convert hardware output. Scaled by 256 (e_a) * 16 (mantisa) = 4096
                        hw_f  = real'(out_data) / 4096.0; 
                        
                        error = (hw_f > exp_f) ? (hw_f - exp_f) : (exp_f - hw_f); // Absolute error
                        
                        if (error > CHECK_TOLERANCE) begin
                            $error("[EXP FAIL] HW=%f, Expected=%f, Err=%f (Exceeds %f)", 
                                   hw_f, exp_f, error, CHECK_TOLERANCE);
                            test_passed = 0;
                        end
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------
    task automatic generate_random_vector(output logic signed [DATA_WIDTH-1:0] v [NUM_INPUTS], input logic mode);
        if (mode == 1'b0) begin
            // Standard full signed range for MAC
            foreach (v[i]) v[i] = $urandom(); 
        end else begin
            // Strictly negative Q4.4 values for EXP (0 down to -128)
            // -120 / 16.0 = --8 (Full supported hardware range)
            foreach (v[i]) v[i] = -($urandom_range(0, 128));
        end
    endtask

endmodule