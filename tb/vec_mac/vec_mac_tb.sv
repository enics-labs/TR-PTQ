module vec_mac_tb;

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    localparam int DATA_WIDTH = 8;
    localparam int NUM_INPUTS = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int NUM_TV     = 10;
    localparam int LATENCY    = 3; // vec_mac_dsp48 pipeline depth

    // ------------------------------------------------------------
    // Signals
    // ------------------------------------------------------------
    logic clk;
    logic rst_n;

    logic valid_in;
    logic in_ready;
    logic clear_acc;

    logic out_valid;
    logic out_ready;

    logic signed [DATA_WIDTH-1:0] vec1 [NUM_INPUTS];
    logic signed [DATA_WIDTH-1:0] vec2 [NUM_INPUTS];
    logic signed [ACC_WIDTH-1:0]  out_data;

    // ------------------------------------------------------------
    // Expected model
    // ------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] expected_acc;
    logic signed [ACC_WIDTH-1:0] expected_queue [$];

    int test_passed;

    // ------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------
    vec_mac_dsp48 #(
        .N(NUM_INPUTS),
        .W(DATA_WIDTH),
        .ACC_W(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(valid_in),
        .in_ready(in_ready),
        .a(vec1),
        .b(vec2),
        .clear_acc(clear_acc),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_dot(out_data)
    );

    // ------------------------------------------------------------
    // Clock
    // ------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ------------------------------------------------------------
    // Stimulus
    // ------------------------------------------------------------
    initial begin
        test_passed = 1;
        out_ready   = 1'b1;
        valid_in    = 0;
        clear_acc   = 0;
        rst_n       = 0;
        expected_acc = 0;

        // Reset
        repeat (3) @(posedge clk);
        rst_n = 1;

        // --------------------------------------------------------
        // First accumulation window
        // --------------------------------------------------------
        for (int t = 0; t < NUM_TV; t++) begin
            push_vector(t == 0); // clear_acc on first
        end

        // --------------------------------------------------------
        // Second accumulation window
        // --------------------------------------------------------
        for (int t = 0; t < NUM_TV; t++) begin
            push_vector(t == 0);
        end

        // Drain pipeline
        repeat (LATENCY + 2) @(posedge clk);

        if (test_passed)
            $display("\n\033[0;32m[TEST PASSED]\033[0m");
        else
            $display("\n\033[0;31m[TEST FAILED]\033[0m");

        $finish;
    end

    // ------------------------------------------------------------
    // Push one vector
    // ------------------------------------------------------------
    task push_vector(input bit first);
        logic signed [DATA_WIDTH-1:0] v1 [NUM_INPUTS];
        logic signed [DATA_WIDTH-1:0] v2 [NUM_INPUTS];
        logic signed [ACC_WIDTH-1:0] dot;

        generate_random_vector(v1);
        generate_random_vector(v2);

        dot = 0;
        for (int i = 0; i < NUM_INPUTS; i++)
            dot += v1[i] * v2[i];

        if (first)
            expected_acc = dot;
        else
            expected_acc += dot;

        expected_queue.push_back(expected_acc);

        @(posedge clk);
        valid_in  <= 1'b1;
        clear_acc <= first;
        vec1      <= v1;
        vec2      <= v2;

        @(posedge clk);
        valid_in  <= 0;
        clear_acc <= 0;
    endtask

    // ------------------------------------------------------------
    // Checker
    // ------------------------------------------------------------
    initial begin
        logic signed [ACC_WIDTH-1:0] exp;

        repeat (LATENCY) @(posedge clk);

        forever begin
            @(posedge clk);
            if (out_valid) begin
                if (expected_queue.size() == 0) begin
                    $error("Unexpected output!");
                    test_passed = 0;
                end else begin
                    exp = expected_queue.pop_front();
                    if (out_data !== exp) begin
                        $error("Mismatch! Expected=%0d Got=%0d", exp, out_data);
                        test_passed = 0;
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------
    task generate_random_vector(output logic signed [DATA_WIDTH-1:0] v [NUM_INPUTS]);
        int min = -(1 << (DATA_WIDTH-1));
        int max =  (1 << (DATA_WIDTH-1)) - 1;
        foreach (v[i])
            v[i] = $urandom_range(max, min);
    endtask

endmodule
