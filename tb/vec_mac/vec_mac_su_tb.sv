// ===================================================================================
// TESTBENCH ARCHITECTURE: Vector MAC with Mixed-Sign Support (vec_mac_su_tb)
// ===================================================================================
// This self-checking testbench verifies the `vec_mac_dsp48` module across all 
// three of its operational modes: Signed x Signed (0), Signed x Unsigned (1), 
// and Unsigned x Unsigned (2).
//
// 1. THE STIMULUS:
//    The testbench generates random 8-bit raw logic vectors (`vec1`, `vec2`) and 
//    pushes them into the hardware pipeline over multiple clock cycles to simulate
//    a continuous accumulation window.
//
// 2. THE GOLDEN MATH MODEL (push_vector task):
//    To accurately predict the hardware's output, the testbench computes the 
//    expected dot product in software. Crucially, it dynamically mimics the 
//    hardware's exact casting and zero-padding logic based on the `op_mode`:
//      * Mode 0 (SS): Casts both raw vectors directly to $signed().
//      * Mode 1 (SU): Zero-pads the second vector `{1'b0, v2}` to force a positive 
//                     unsigned magnitude before the $signed() cast.
//      * Mode 2 (UU): Zero-pads both vectors before the $signed() cast.
//
// 3. PIPELINE & SELF-CHECKING:
//    Because the hardware is a pipelined DSP48-style MAC, it has a fixed latency 
//    (`LATENCY = 3`). The testbench pushes the golden expected sums into a FIFO 
//    queue (`expected_queue`). A background checker block monitors the output bus, 
//    waits for `out_valid` to assert, pops the oldest expected value, and strictly 
//    compares it against the hardware's `out_data`.
// ===================================================================================
module vec_mac_su_tb;

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    localparam int DATA_WIDTH = 8;
    localparam int NUM_INPUTS = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int NUM_TV     = 5;
    localparam int LATENCY    = 3; // vec_mac_dsp48 pipeline depth

    // ------------------------------------------------------------
    // Signals
    // ------------------------------------------------------------
    logic clk;
    logic rst_n;

    logic valid_in;
    logic in_ready;
    logic clear_acc;
    logic [1:0] op_mode; // 0:SS, 1:SU, 2:UU

    logic out_valid;
    logic out_ready;

    logic [DATA_WIDTH-1:0] vec1 [NUM_INPUTS];
    logic [DATA_WIDTH-1:0] vec2 [NUM_INPUTS];
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
        .op_mode(op_mode),
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

        $display("\n--- Testing Mode 0: Signed x Signed ---");
        for (int t = 0; t < NUM_TV; t++) push_vector(t == 0, 2'd0);

        $display("--- Testing Mode 1: Signed x Unsigned ---");
        for (int t = 0; t < NUM_TV; t++) push_vector(t == 0, 2'd1);

        $display("--- Testing Mode 2: Unsigned x Unsigned ---");
        for (int t = 0; t < NUM_TV; t++) push_vector(t == 0, 2'd2);

        // Drain pipeline
        repeat (LATENCY + 4) @(posedge clk);

        if (test_passed)
            $display("\n\033[0;32m[TEST PASSED]\033[0m");
        else
            $display("\n\033[0;31m[TEST FAILED]\033[0m");

        $finish;
    end

    // ------------------------------------------------------------
    // Push one vector and model the math
    // ------------------------------------------------------------
    task automatic push_vector(input bit first, input logic [1:0] mode);
        logic [DATA_WIDTH-1:0] v1 [NUM_INPUTS];
        logic [DATA_WIDTH-1:0] v2 [NUM_INPUTS];
        logic signed [ACC_WIDTH-1:0] dot;

        logic signed [ACC_WIDTH-1:0] ext_v1;
        logic signed [ACC_WIDTH-1:0] ext_v2;

        generate_random_vector(v1);
        generate_random_vector(v2);

        dot = 0;
        for (int i = 0; i < NUM_INPUTS; i++) begin
            // Mimic the hardware casting logic exactly
            case (mode)
                2'd0: begin // SS
                    ext_v1 = $signed(v1[i]);
                    ext_v2 = $signed(v2[i]);
                end
                2'd1: begin // SU
                    ext_v1 = $signed(v1[i]);
                    ext_v2 = $signed({1'b0, v2[i]}); // Force positive
                end
                default: begin // UU
                    ext_v1 = $signed({1'b0, v1[i]}); // Force positive
                    ext_v2 = $signed({1'b0, v2[i]}); // Force positive
                end
            endcase
            dot += (ext_v1 * ext_v2);
        end

        if (first)
            expected_acc = dot;
        else
            expected_acc += dot;

        expected_queue.push_back(expected_acc);

        @(posedge clk);
        valid_in  <= 1'b1;
        clear_acc <= first;
        op_mode   <= mode;
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
                    $error("Unexpected output! Got=%0d", out_data);
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
    task automatic generate_random_vector(output logic [DATA_WIDTH-1:0] v [NUM_INPUTS]);
        int min = -(1 << (DATA_WIDTH-1));
        int max =  (1 << (DATA_WIDTH-1)) - 1;
        foreach (v[i])
            v[i] = $urandom_range(max, min);
    endtask

endmodule
