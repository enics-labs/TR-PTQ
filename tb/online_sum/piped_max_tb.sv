module tb_pipelined_max();

    // Parameters
    localparam int NUM_INPUTS = 8;
    localparam int DATA_WIDTH = 8;
    localparam int NUM_OF_TV = 8;
    localparam int LATENCY    = $clog2(NUM_INPUTS);
    // Define color codes as localparams
    localparam string GREEN = "\033[0;32m";
    localparam string RED   = "\033[0;31m";
    localparam string RESET = "\033[0m";

    // Signals
    logic clk;
    int test_passed;

    logic rst_n;
    logic valid_in;
    logic valid_out;
    logic signed [DATA_WIDTH-1:0] in_data [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] max_out;

    // Expected value queue to handle pipeline latency
    logic signed [DATA_WIDTH-1:0] expected_queue [$];

    // Instantiate the Unit Under Test (UUT)
    pipelined_max_tree #(
        .NUM_INPUTS(NUM_INPUTS),
        .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(in_data),
        .valid_in(valid_in),
        .valid_out(valid_out),
        .max_out(max_out)
    );

    // Clock generation (100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Stimulus process
    initial begin
        // Initialize
        init_sim();
        repeat(NUM_OF_TV) push_test_vector();
        end_sim();
    end

    
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

    // Task to apply inputs and calculate expected max
    task  push_test_vector();
        logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS];
        logic signed [DATA_WIDTH-1:0] current_max;
        generate_random_vector(.vec(vec));

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

    // Checker process: Compare output with queue after latency
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

    task print_vector_clean(input logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS]);
        foreach (vec[i]) begin
            $write("%d ", vec[i]); // %d for decimal, %h for hex
        end
        $write("\n"); // Move to next line after the loop
    endtask

    // Task to fill a vector with random signed values
    task generate_random_vector(output logic signed [DATA_WIDTH-1:0] vec [NUM_INPUTS]);
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