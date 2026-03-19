// ===================================================================================
// TESTBENCH ARCHITECTURE: Pipelined Max Tree (piped_max_tb)
// ===================================================================================
// This self-checking testbench verifies the `pipelined_max_tree` module, which 
// computes the maximum value of an N-element vector over a log2(N) stage pipeline.
//
// 1. THE GOLDEN MATH MODEL & PIPELINE TRACKING:
//    The testbench uses a software-based golden model (a standard `for` loop) to 
//    find the true maximum of the generated stimulus vector. Because the hardware 
//    has a multi-cycle latency, the golden result is pushed into an `expected_queue`.
//    A background checker monitors `valid_out`, pops the oldest expected value, 
//    and strictly compares it against `max_out`.
//
// 2. VERIFICATION ENHANCEMENTS (Coverage & Edge Cases):
//    To ensure the comparator tree is robust against physical and logical boundaries, 
//    the stimulus is broken into distinct phases:
//
//    * PHASE 1: Directed Edge Cases
//      - Minimum Bounds: All elements set to -128 (tests negative floor).
//      - Maximum Bounds: All elements set to +127 (tests positive saturation).
//      - Spatial Routing: Max value explicitly placed at index 0 and index N-1 
//        to prove the tree correctly routes values from the extreme edges.
//
//    * PHASE 2: Pipeline Saturation (Back-to-Back Testing)
//      - Asserts `valid_in = 1` continuously while pushing new vectors on every 
//        clock cycle. This proves the internal pipeline stage registers isolate 
//        data correctly and do not suffer from data-collision or overwrite bugs.
//
// 3. SYSTEMVERILOG BEST PRACTICES:
//    All stimulus generation tasks (`push_test_vector`, `generate_random_vector`, etc.) 
//    are explicitly marked as `automatic`. This ensures local variables are dynamically 
//    allocated per call, preventing variable-sharing "ghost bugs" when tasks are 
//    called rapidly back-to-back during Phase 2.
// ===================================================================================
module piped_max_tb();

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    localparam int NUM_INPUTS = 8;
    localparam int DATA_WIDTH = 8;
    localparam int NUM_OF_TV = 8;
    localparam int LATENCY    = $clog2(NUM_INPUTS);
    // Define color codes as localparams
    localparam string GREEN = "\033[0;32m";
    localparam string RED   = "\033[0;31m";
    localparam string RESET = "\033[0m";

    // ------------------------------------------------------------
    // Signals
    // ------------------------------------------------------------
    logic clk;
    int test_passed;

    logic rst_n;
    logic valid_in;
    logic valid_out;
    logic signed [DATA_WIDTH-1:0] in_data [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] max_out;

    // ------------------------------------------------------------
    // Expected Value Queue for the Pipeline
    // ------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] expected_queue [$];

    // ------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------
    piped_max #(
        .NUM_INPUTS(NUM_INPUTS),
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(in_data),
        .valid_in(valid_in),
        .valid_out(valid_out),
        .max_out(max_out)
    );

    // ------------------------------------------------------------
    // Clock (100MHz) & Initialization
    // ------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Stimulus process
    initial begin
        logic signed [DATA_WIDTH-1:0] directed_vec [NUM_INPUTS];
        init_sim();

        $display("\n--- PHASE 1: Edge Cases ---");
        // Edge Case 1: All lowest negative
        for(int i=0; i<NUM_INPUTS; i++) directed_vec[i] = -128;
        push_directed_vector(directed_vec);

        // Edge Case 2: All highest positive
        for(int i=0; i<NUM_INPUTS; i++) directed_vec[i] = 127;
        push_directed_vector(directed_vec);
        
        // Edge Case 3: Max at index 0, rest negative
        for(int i=0; i<NUM_INPUTS; i++) directed_vec[i] = -50;
        directed_vec[0] = 100;
        push_directed_vector(directed_vec);

        // Edge Case 4: Max at last index
        for(int i=0; i<NUM_INPUTS; i++) directed_vec[i] = -50;
        directed_vec[NUM_INPUTS-1] = 100;
        push_directed_vector(directed_vec);

        $display("\n--- PHASE 2: Pipeline Saturation (Back-to-Back) ---");
        // Do not drop valid_in between pushes!
        repeat(10) push_test_vector_saturated();

        // Drop valid_in after the saturation burst
        @(posedge clk);
        valid_in = 1'b0;

        end_sim();
    end

    // ------------------------------------------------------------
    // Init+End & Push Vector Tasks
    // ------------------------------------------------------------
    task init_sim();
        test_passed = 1;

        rst_n = 0;
        valid_in = 0;
        for (int i = 0; i < NUM_INPUTS; i++) in_data[i] = 0;
        
        repeat(2) @(posedge clk);
        rst_n = 1;
        @(posedge clk);        
    endtask

    task end_sim();
        // Add dummy data to push the last real test case through the pipe
        // repeat(LATENCY) push_test_vector('{default: 0});
        repeat(LATENCY) @(posedge clk);
        #50;
        if (test_passed) begin
            $display("%s[TEST PASSED]%s", GREEN, RESET);
        end else begin
            $display("%s[TEST FAILED]%s", RED, RESET);
        end    
        $finish;
    endtask 

    task automatic push_directed_vector(input logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS]);
        logic signed [DATA_WIDTH-1:0] current_max;
        in_data = vec;
        valid_in = 1'b1;
        
        // Calculate expected max manually
        current_max = vec[0];
        for (int i = 1; i < NUM_INPUTS; i++) begin
            if (vec[i] > current_max) current_max = vec[i];
        end
        
        print_vector_clean(vec);
        expected_queue.push_back(current_max);
        @(posedge clk);
        valid_in = 1'b0;
    endtask

    task automatic push_test_vector_saturated();
        logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS];
        logic signed [DATA_WIDTH-1:0] current_max;
        generate_random_vector(vec);

        in_data = vec;
        valid_in = 1'b1;
        
        current_max = vec[0];
        for (int i = 1; i < NUM_INPUTS; i++) begin
            if (vec[i] > current_max) current_max = vec[i];
        end
        
        print_vector_clean(vec);
        expected_queue.push_back(current_max);
        @(posedge clk); // Advance clock, but keep valid_in HIGH
    endtask

    // ------------------------------------------------------------
    // Background Checker
    // ------------------------------------------------------------
    initial begin
        logic signed [DATA_WIDTH-1:0] expected_val;
        
        // Wait for the pipeline to fill
        repeat(LATENCY + 1) @(posedge clk);

        forever begin
            @(posedge clk);
            if (valid_out) begin
                if (expected_queue.size() > 0) begin
                    expected_val = expected_queue.pop_front();
                    if (max_out !== expected_val) begin
                        $error("Mismatch! Time=%0t | Expected=%d | Got=%d", $time, expected_val, max_out);
                        test_passed = 0;
                    end else begin
                        $display("Success! Time=%0t | Expected=%d | Got=%d", $time, expected_val, max_out);
                    end
                end else begin
                    $error("Valid out with empty Q");
                end                
            end
        end
    end

    // ------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------
    task automatic print_vector_clean(input logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS]);
        foreach (vec[i]) begin
            $write("%d ", vec[i]); // %d for decimal, %h for hex
        end
        $write("\n"); // Move to next line after the loop
    endtask

    // Task to fill a vector with random signed values
    task automatic generate_random_vector(output logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS]);
        // Calculate min/max for signed range
        // For 8 bits: min = -128, max = 127
        longint min_val = -(1 << (DATA_WIDTH-1));
        longint max_val = (1 << (DATA_WIDTH-1)) - 1;

        foreach (vec[i]) begin
            // Generate a random value within the signed boundaries
            vec[i] = $urandom_range(max_val, min_val);
        end
    endtask
endmodule