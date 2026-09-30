`timescale 1ns/1ps

module symmetry_modifier_tb();

    // Parameters matches standard Q4.4 precision
    localparam int N       = 4;
    localparam int WIDTH_X = 8;
    localparam int WIDTH_Y = 8;
    localparam int FRAC_W  = 4;

    // Signals
    logic signed [WIDTH_X-1:0] x_raw [N];
    logic signed [WIDTH_Y-1:0] y_sig [N];
    logic                      mode_en;
    logic signed [WIDTH_Y-1:0] sig_corrected [N];

    // DUT Instantiation
    symmetry_modifier #(
        .N(N), .WIDTH_X(WIDTH_X), .WIDTH_Y(WIDTH_Y), .FRAC_W(FRAC_W)
    ) dut (
        .x_raw(x_raw),
        .y_sig(y_sig),
        .mode_en(mode_en),
        .sig_corrected(sig_corrected)
    );

    // Task: Check Output Array
    task check_output(
        input int expected [N],
        input string test_name
    );
        logic pass = 1'b1;
        for (int i = 0; i < N; i++) begin
            if (sig_corrected[i] !== expected[i]) begin
                $error("[FAIL] %s Lane %0d | Expected: %0d, Got: %0d", 
                        test_name, i, expected[i], sig_corrected[i]);
                pass = 1'b0;
            end
        end
        if (pass) $display("[PASS] %s", test_name);
    endtask

    // Main Test Sequence
    initial begin
        $display("Starting symmetry_modifier Tests...");
        
        // ----------------------------------------------------
        // Test 1: Bypass Mode
        // ----------------------------------------------------
        mode_en = 0;
        x_raw = '{-10, 20, -30, 40}; // Raw signs should be ignored
        y_sig = '{5, 6, 7, 8};
        #10;
        check_output('{5, 6, 7, 8}, "Bypass Mode");

        // ----------------------------------------------------
        // Test 2: Enabled, Mixed Signs
        // ----------------------------------------------------
        // ONE_Q = 16 (1 << 4).
        // Lane 0: x < 0  -> 16 - 5 = 11
        // Lane 1: x >= 0 -> 6 (unchanged)
        // Lane 2: x < 0  -> 16 - 7 = 9
        // Lane 3: x >= 0 -> 8 (unchanged)
        mode_en = 1;
        x_raw = '{-10, 20, -5, 100};
        y_sig = '{5, 6, 7, 8};
        #10;
        check_output('{11, 6, 9, 8}, "Symmetry Mode (Mixed Signs)");

        // ----------------------------------------------------
        // Test 3: Edge Cases (All negative raw inputs)
        // ----------------------------------------------------
        // Lane 0: y=16 (1.0)  -> 16 - 16 = 0  (0.0)
        // Lane 1: y=0  (0.0)  -> 16 - 0  = 16 (1.0)
        // Lane 2: y=8  (0.5)  -> 16 - 8  = 8  (0.5)
        // Lane 3: y=12 (0.75) -> 16 - 12 = 4  (0.25)
        mode_en = 1;
        x_raw = '{-1, -1, -1, -1}; 
        y_sig = '{16, 0, 8, 12};
        #10;
        check_output('{0, 16, 8, 4}, "Symmetry Mode (Edge Cases)");

        #20;
        $display("ALL SYMMETRY MODIFIER TESTS PASSED!");
        $finish;
    end
endmodule