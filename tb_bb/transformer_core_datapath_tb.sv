`timescale 1ns/1ps

module transformer_core_datapath_tb();

    localparam int N          = 8;
    localparam int W          = 8;
    localparam int ACC_W      = 32;
    localparam int FRAC       = 4;

    // ========================================================================
    // CLOCK & RESET
    // ========================================================================
    logic clk;
    logic rst_n;

    always #5 clk = ~clk; // 100MHz Clock

    // ========================================================================
    // DUT SIGNALS
    // ========================================================================
    logic                    in_valid;
    logic signed [W-1:0]     a [N];
    logic signed [W-1:0]     b [N];
    
    logic                    out_valid;
    logic signed [ACC_W-1:0] out_vec [N];
    logic signed [ACC_W-1:0] out_dot;

    logic                    ctrl_tr_lane0_mode;
    logic [1:0]              ctrl_tr_shift_mode;
    logic                    ctrl_tr_exp_sel;
    logic                    ctrl_mux_tr_vec_sel;
    logic [1:0]              ctrl_mux_a_sel;
    logic [1:0]              ctrl_mux_b_sel;
    logic [1:0]              ctrl_mac_op_mode;
    logic                    ctrl_mac_elemwise;
    logic                    ctrl_mac_clear_acc;
    logic                    ctrl_mac_in_valid;
    logic                    ctrl_save_sum;

    // ========================================================================
    // DUT INSTANTIATION
    // ========================================================================
    transformer_core_datapath #(
        .N(N), .W(W), .ACC_W(ACC_W), .FRAC(FRAC)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .a(a), .b(b),
        .out_valid(out_valid), .out_vec(out_vec), .out_dot(out_dot),
        
        .ctrl_tr_lane0_mode(ctrl_tr_lane0_mode),
        .ctrl_tr_shift_mode(ctrl_tr_shift_mode),
        .ctrl_tr_exp_sel(ctrl_tr_exp_sel),
        
        .ctrl_mux_tr_vec_sel(ctrl_mux_tr_vec_sel),
        .ctrl_mux_a_sel(ctrl_mux_a_sel),
        .ctrl_mux_b_sel(ctrl_mux_b_sel),
        
        .ctrl_mac_op_mode(ctrl_mac_op_mode),
        .ctrl_mac_elemwise(ctrl_mac_elemwise),
        .ctrl_mac_clear_acc(ctrl_mac_clear_acc),
        .ctrl_mac_in_valid(ctrl_mac_in_valid),
        .ctrl_save_sum(ctrl_save_sum)
    );

    // ========================================================================
    // MOCK FSM TASKS (The Control Sequences)
    // ========================================================================
    task reset_controls();
        in_valid            = 0;
        ctrl_tr_lane0_mode  = 0;
        ctrl_tr_shift_mode  = 0;
        ctrl_tr_exp_sel     = 0;
        ctrl_mux_tr_vec_sel = 0;
        ctrl_mux_a_sel      = 0;
        ctrl_mux_b_sel      = 0;
        ctrl_mac_op_mode    = 0;
        ctrl_mac_elemwise   = 0;
        ctrl_mac_clear_acc  = 0;
        ctrl_mac_in_valid   = 0; // Keep MAC gated by default
        ctrl_save_sum       = 0;
    endtask

    // ---------------------------------------------------------
    // INSTRUCTION: Linear Matrix-Vector Multiply
    // ---------------------------------------------------------
    task issue_linear_mvm();
        ctrl_mux_a_sel      = 2'b00; // Mem A
        ctrl_mux_b_sel      = 2'b00; // Mem B
        ctrl_mac_elemwise   = 1'b0;  // Dot Product Mode
        ctrl_mac_clear_acc  = 1'b1;  // Clear MAC accumulators for fresh sum
        ctrl_mac_op_mode    = 2'b00; // Signed-Signed (SS) Mode
        ctrl_tr_lane0_mode  = 1'b0;
    endtask

    // ---------------------------------------------------------
    // INSTRUCTION: SoftMax Phase 1 (Calculates denominator)
    // ---------------------------------------------------------
    task issue_softmax_phase1();
        ctrl_mux_tr_vec_sel = 1'b0;  // Route Max_Sub into TR Array
        ctrl_tr_exp_sel     = 1'b0;  // Bypass TR_LN (pure exp mode)
        ctrl_tr_lane0_mode  = 1'b0;  // All lanes in Vector Mode
        
        ctrl_mux_a_sel      = 2'b01; // Route TR Anchor (e_a) into MAC A
        ctrl_mux_b_sel      = 2'b01; // Route TR Mantissa into MAC B
        
        ctrl_mac_elemwise   = 1'b0;  // Element-wise multiply to reconstruct e^x
        ctrl_mac_clear_acc  = 1'b1;  // Clear MAC accumulators for a fresh sum
        
        ctrl_mac_op_mode    = 2'b10; // Unsigned-Unsigned (UU) Mode
    endtask

    // ---------------------------------------------------------
    // INSTRUCTION: SoftMax Phase 2 (Calculates numerators)
    // ---------------------------------------------------------
    task issue_softmax_phase2();
        ctrl_tr_lane0_mode  = 1'b0;
        ctrl_tr_exp_sel     = 1'b0;
        ctrl_tr_lane0_mode  = 1'b0;
        ctrl_mux_a_sel      = 2'b01; // Buffered Exponentials
        ctrl_mux_b_sel      = 2'b01;
        ctrl_mac_elemwise   = 1'b1;  // ELEMENT-WISE MODE
        ctrl_mac_clear_acc  = 1'b1;  
        ctrl_mac_op_mode    = 2'b10; // UU Mode
    endtask

    // ---------------------------------------------------------
    // INSTRUCTION: SoftMax Phase 3 (Final Probabilities)
    // ---------------------------------------------------------
    task issue_softmax_phase3();
        ctrl_tr_lane0_mode  = 1'b1;
        ctrl_tr_shift_mode  = 2'b01;
        ctrl_tr_exp_sel     = 1'b1;
        
        ctrl_mux_a_sel      = 2'b10; // Route buffered out_vec into MAC A
        ctrl_mux_b_sel      = 2'b10; // Broadcast the Lane 0 scalar to MAC B
        
        ctrl_mac_elemwise   = 1'b1;  // Element-wise multiply for final probabilities
        ctrl_mac_clear_acc  = 1'b1;  

        ctrl_mac_op_mode    = 2'b10; // Unsigned-Unsigned (UU) Mode
    endtask

    // ---------------------------------------------------------
    // INSTRUCTION: GELU Phase 1 (Compute CDF)
    // ---------------------------------------------------------
    task issue_gelu_phase1();
        ctrl_mux_tr_vec_sel = 1'b1;  // Route RAW Delayed A (Bypass Max_Sub)
        ctrl_tr_exp_sel     = 1'b0;  
        ctrl_tr_lane0_mode  = 1'b0;  // All lanes independent
        
        ctrl_mux_a_sel      = 2'b01; // TR Anchor
        ctrl_mux_b_sel      = 2'b01; // TR Mantissa
        
        ctrl_mac_elemwise   = 1'b1;  // Element-wise multiply
        ctrl_mac_clear_acc  = 1'b1;  
        ctrl_mac_op_mode    = 2'b10; // UU Mode (CDF is strictly positive)
    endtask

    // ---------------------------------------------------------
    // INSTRUCTION: GELU Phase 2 (Multiply x * CDF)
    // ---------------------------------------------------------
    task issue_gelu_phase2();
        ctrl_mux_a_sel      = 2'b10; // Buffered CDF from out_vec
        ctrl_mux_b_sel      = 2'b00; // Raw Memory B (We will feed X here!)
        
        ctrl_mac_elemwise   = 1'b1;  // Element-wise multiply
        ctrl_mac_clear_acc  = 1'b1;  
        ctrl_mac_op_mode    = 2'b00; // SS Mode (Raw X can be negative)
    endtask

    // ========================================================================
    // MAIN VERIFICATION SEQUENCE
    // ========================================================================
    initial begin
        clk = 0;
        rst_n = 0;
        reset_controls();
        for (int i=0; i<N; i++) begin a[i] = '0; b[i] = '0; end

        $display("\n=======================================================================");
        $display(" STARTING DATAPATH WRAPPER VERIFICATION (OPTION A + MAC UU MODE)");
        $display("=======================================================================\n");

        #20 rst_n = 1;
        #10;

        // ====================================================================
        // TEST 1: LINEAR BYPASS (MVM)
        // ====================================================================
        $display(">>> TEST 1: Pure Linear Datapath (Bypassing Non-Linear Math)");
        issue_linear_mvm();
        
        for (int i=0; i<N; i++) begin 
            a[i] = 8'd2; // 2.0 
            b[i] = 8'd3; // 3.0 
        end
        in_valid = 1;
        
        // Wait exactly 3 clocks for the `piped_max` delay lines to align the data
        #30; 
        
        // Data is now sitting at the MAC MUX inputs. Open the MAC gate!
        ctrl_mac_in_valid = 1;
        #10; // Pulse for 1 clock cycle
        
        // Close gates
        in_valid = 0;
        ctrl_mac_in_valid = 0;
        ctrl_mac_clear_acc = 0;
        
        #30; // Wait for MAC to settle
        $display("   Expected Accumulator Sum: 8 lanes * (2 * 3) = 48");
        $display("   Hardware MAC out_dot  : %0d", out_dot);
        if (out_dot == 48) $display("   [PASS] Linear Datapath is perfectly clean!");
        else               $display("   [FAIL] Data corruption in multiplexers.");

        #30;

        // ====================================================================
        // TEST 2: SOFTMAX MULTI-CYCLE ORCHESTRATION
        // ====================================================================
        $display(">>> TEST 2: SoftMax Multi-Cycle Routing (The Ultimate Test)");
        
        a[0] =  8'd0;    // Max value
        a[1] = -8'd16;   // -1.0
        a[2] = -8'd32;   // -2.0
        for (int i=3; i<N; i++) a[i] = -8'd128; // Flush to ~0

        // --- CYCLE 1: SUMMATION ---
        issue_softmax_phase1();
        in_valid = 1;
        #30; // Wait 3 clocks for Piped Max Tree to align
        
        ctrl_mac_in_valid = 1;
        #10; // Pulse for 1 clock cycle
        
        in_valid = 0;
        ctrl_mac_in_valid = 0;
        ctrl_mac_clear_acc = 0;
        #30; // Wait for MAC to accumulate and settle
        ctrl_save_sum = 1; 
        #10;
        ctrl_save_sum = 0;
        $display("\n   [Cycle 1] Vector Sum Computed in out_dot: %f", real'(out_dot) / 4096.0);
        
        // --- CYCLE 2: RECOMPUTE EXPONENTIALS ---
        issue_softmax_phase2();
        
        ctrl_mac_in_valid = 1;
        #10; 
        
        ctrl_mac_in_valid = 0;
        ctrl_mac_clear_acc = 0;
        #30; 
        
        $display("\n   [Cycle 2] Vectors buffered securely in out_vec!");

        // --- CYCLE 3: BROADCAST DIVIDE ---
        issue_softmax_phase3();
        #10;

        ctrl_mac_in_valid = 1;
        #10; 

        ctrl_mac_in_valid = 0;
        ctrl_mac_clear_acc = 0;
        #30; // Wait for MAC to settle
        
        $display("\n   [Cycle 3] Final SoftMax Probabilities:");
        // Format is Q4.4 * Q4.4 = Q8.8
        $display("      Lane 0 Prob: %f", real'(out_vec[0]) / 256.0); 
        $display("      Lane 1 Prob: %f", real'(out_vec[1]) / 256.0);
        $display("      Lane 2 Prob: %f", real'(out_vec[2]) / 256.0);

        // ====================================================================
        // TEST 3: GELU 2-CYCLE RECOMPUTE
        // ====================================================================
        $display("\n>>> TEST 3: GELU 2-Cycle Recompute Architecture");
        
        // Setup inputs (in Q4.4 format)
        a[0] =  8'd0;    //  0.0
        a[1] =  8'd16;   //  1.0
        a[2] = -8'd16;   // -1.0
        for (int i=3; i<N; i++) a[i] = -8'd128; // Flush to ~0

        // --- CYCLE 1: COMPUTE CDF (Phi(x)) ---
        issue_gelu_phase1();
        in_valid = 1;
        #30; // Wait 3 clocks for the delay lines
        
        ctrl_mac_in_valid = 1;
        #10; // Pulse MAC Valid
        
        in_valid = 0;
        ctrl_mac_in_valid = 0;
        ctrl_mac_clear_acc = 0;
        #30; // Wait for MAC pipeline to settle
        
        $display("   [Cycle 1] CDFs (Phi(x)) buffered securely in out_vec!");

        // --- CYCLE 2: MULTIPLY x * CDF ---
        issue_gelu_phase2();
        
        // RECOMPUTE: Feed the exact same vector, but map it to MUX B!
        for (int i=0; i<N; i++) b[i] = a[i]; 
        in_valid = 1; 
        #30; // Wait 3 clocks for the delay lines
        
        ctrl_mac_in_valid = 1;
        #10; // Pulse MAC Valid
        
        in_valid = 0;
        ctrl_mac_in_valid = 0;
        ctrl_mac_clear_acc = 0;
        #30; // Wait for MAC pipeline to settle
        
        $display("\n   [Cycle 2] Final GELU Activations:");
        // Format is Q4.4 (CDF) * Q4.4 (X) = Q8.8
        $display("      Lane 0 (x= 0.0): %f", real'(out_vec[0]) / 256.0); 
        $display("      Lane 1 (x= 1.0): %f", real'(out_vec[1]) / 256.0);
        $display("      Lane 2 (x=-1.0): %f", real'(out_vec[2]) / 256.0);
        
        $display("=======================================================================\n");
        $finish;
    end

endmodule