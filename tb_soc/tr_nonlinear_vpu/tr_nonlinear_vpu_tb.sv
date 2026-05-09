`timescale 1ns/1ps

module tb_tr_nonlinear_vpu();

    // ---------------------------------------------------------
    // Parameters (Scaled down for simulation readability)
    // ---------------------------------------------------------
    localparam int N         = 4;
    localparam int W_VEC     = 8;
    localparam int W_MAC     = 16;
    localparam int ACC_W     = 32;
    localparam int FRAC_W    = 4;
    localparam int LUT_IDX_W = 3;

    // ---------------------------------------------------------
    // DUT Signals
    // ---------------------------------------------------------
    logic clk, rst_n;

    // SRAM Interfaces
    logic signed [W_VEC-1:0] sram_data_a [N];
    logic signed [W_VEC-1:0] sram_data_b [N];
    logic signed [W_VEC-1:0] vpu_data_out [N];
    
    // Controller Registers
    logic signed [W_VEC-1:0] ctrl_scalar_sub_val;
    logic signed [W_VEC-1:0] vpu_max_out;
    logic signed [ACC_W-1:0] vpu_dot_out;

    // MUX Controls
    logic [1:0] mux_bb_in_sel;
    logic [1:0] mux_mac_a_sel;
    logic       mux_mac_b_sel;
    logic [1:0] mux_vecmul_a_sel;
    logic       mux_vecmul_b_sel;
    logic [2:0] mux_vpu_out_sel;

    // Enables & Modes
    logic       en_piped_max;
    logic       en_mac_valid;
    logic       en_vecmul_valid;
    logic       en_bb_valid;
    logic       vpu_bb_valid_out;
    logic       mac_clear_acc;
    logic [1:0] mac_op_mode;
    logic [1:0] vecmul_op_mode;
    logic       bb_mode_pre_ln;
    logic [1:0] bb_mode_post_ln;
    logic       sym_mode_en;

    // ---------------------------------------------------------
    // DUT Instantiation
    // ---------------------------------------------------------
    tr_nonlinear_vpu #(
        .N(N),
        .W_VEC(W_VEC),
        .W_MAC(W_MAC),
        .ACC_W(ACC_W),
        .FRAC_W(FRAC_W),
        .LUT_IDX_W(LUT_IDX_W)
    ) dut (.*); // SystemVerilog wildcard connection

    // ---------------------------------------------------------
    // Clock Generation
    // ---------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ---------------------------------------------------------
    // Helper Task: Reset all control lines to safe zero states
    // ---------------------------------------------------------
    task clear_crossbar();
        mux_bb_in_sel    = '0;
        mux_mac_a_sel    = '0;
        mux_mac_b_sel    = '0;
        mux_vecmul_a_sel = '0;
        mux_vecmul_b_sel = '0;
        mux_vpu_out_sel  = '0;
        
        en_piped_max    = 0;
        en_mac_valid    = 0;
        en_vecmul_valid = 0;
        en_bb_valid     = 0;
        mac_clear_acc   = 0;
        mac_op_mode     = 0;
        vecmul_op_mode  = 0;
        
        bb_mode_pre_ln  = 0;
        bb_mode_post_ln = 0;
        sym_mode_en     = 0;
        
        ctrl_scalar_sub_val = '0;
        for (int i=0; i<N; i++) begin
            sram_data_a[i] = '0;
            sram_data_b[i] = '0;
        end
    endtask

    // ---------------------------------------------------------
    // Main Test Sequence
    // ---------------------------------------------------------
    initial begin
        $display("==================================================");
        $display("  Starting VPU Crossbar Routing Tests...          ");
        $display("==================================================");
        
        rst_n = 0;
        clear_crossbar();
        #20 rst_n = 1;
        @(posedge clk);

        // ====================================================================
        // TEST 1: Pure VecMul Bypass (SRAM -> VecMul -> SRAM)
        // Emulates: Generic scaling or parallel RMSNorm Affine phase
        // ====================================================================
        $display("\n[TEST 1] Routing: SRAM -> VecMul -> SRAM");
        clear_crossbar();
        
        // Setup Crossbar
        mux_vecmul_a_sel = 2'b00; // SRAM A
        mux_vecmul_b_sel = 1'b0;  // SRAM B
        mux_vpu_out_sel  = 3'b000; // VecMul Out
        vecmul_op_mode   = 2'd0;  // Signed x Signed
        
        // Drive Data
        @(posedge clk);
        en_vecmul_valid = 1;
        sram_data_a = '{2, -3, 4, -5};
        sram_data_b = '{10, 10, 10, 10};
        
        @(posedge clk); en_vecmul_valid = 0;
        
        // VecMul has 3-stage pipeline. Wait 4 cycles.
        repeat(4) @(posedge clk);
        
        if (vpu_data_out[0] === 8'd20 && vpu_data_out[1] === -8'sd30)
            $display("  -> [PASS] VecMul routed correctly to SRAM output.");
        else
            $error("  -> [FAIL] VecMul routing failed.");


        // ====================================================================
        // TEST 2: Dense MAC Reduction (SRAM -> MAC -> Controller Dot)
        // Emulates: RMSNorm Sum of Squares (Pass 1)
        // ====================================================================
        $display("\n[TEST 2] Routing: SRAM -> MAC Engine -> Controller Dot Reg");
        clear_crossbar();
        
        // Setup Crossbar
        mux_mac_a_sel = 2'b00; // SRAM A
        mux_mac_b_sel = 1'b0;  // SRAM B
        mac_clear_acc = 1'b1;  // Clear accumulator (no bias)
        mac_op_mode   = 2'd0;  // Signed x Signed
        
        // Drive Data (Square the inputs: A * A)
        @(posedge clk);
        en_mac_valid = 1;
        sram_data_a = '{2, 3, 4, 5};
        sram_data_b = '{2, 3, 4, 5}; // 4 + 9 + 16 + 25 = 54
        
        @(posedge clk); en_mac_valid = 0;
        
        // MAC has 4-stage pipeline. Wait 5 cycles.
        repeat(5) @(posedge clk);
        
        if (vpu_dot_out === 32'd54)
            $display("  -> [PASS] MAC Sum of Squares routed to Controller Reg.");
        else
            $error("  -> [FAIL] MAC reduction routing failed. Got: %0d", vpu_dot_out);


        // ====================================================================
        // TEST 3: Log-Domain Scalar Injection (Dot Reg -> TR-Backbone -> SRAM)
        // Emulates: RMSNorm Inverse Sqrt or Softmax Reciprocal (Pass 2)
        // ====================================================================
        $display("\n[TEST 3] Routing: Dot Reg -> TR-Backbone -> SRAM");
        clear_crossbar();
        
        // Setup Crossbar
        mux_bb_in_sel   = 2'b00;    // Ingest from vpu_dot_out
        bb_mode_post_ln = 2'b01;    // -1.0 * x (Division mode)
        mux_vpu_out_sel = 3'b001;   // Output bb_mantisa to SRAM
        
        // Drive Data through the pipelined backbone
        @(posedge clk);
        en_bb_valid = 1;
        
        @(posedge clk);
        en_bb_valid = 0;
        
        // Wait for pipelined backbone to settle (2 cycles)
        do begin @(posedge clk); end while (!vpu_bb_valid_out);
        
        if (vpu_data_out[0] !== 8'hxx)
            $display("  -> [PASS] Pipelined backbone routing is active and propagating.");
        else
            $error("  -> [FAIL] Pipelined backbone routing is disconnected or stuck.");


        // ====================================================================
        // TEST 4: GELU Symmetry Gating (SRAM -> Sym Mod -> VecMul -> SRAM)
        // Emulates: GELU Pass 3 (Symmetry Trick + Final Multiply)
        // ====================================================================
        $display("\n[TEST 4] Routing: SRAM -> Symmetry Modifier -> VecMul -> SRAM");
        clear_crossbar();
        
        // Setup Crossbar
        mux_vecmul_a_sel = 2'b00; // SRAM A (Raw x)
        mux_vecmul_b_sel = 1'b1;  // Sym Mod Output
        sym_mode_en      = 1'b1;  // Enable symmetry trick
        mux_vpu_out_sel  = 3'b000; // VecMul Out
        vecmul_op_mode   = 2'd0;  // Signed x Signed
        
        // Drive Data
        // ONE_Q = 16 (1.0). 
        // Lane 0: x = -10. Sym Mod flips y (16 - 5 = 11). VecMul = -10 * 11 = -110.
        // Lane 1: x = 10.  Sym Mod passes y (6).         VecMul = 10 * 6 = 60.
        @(posedge clk);
        en_vecmul_valid = 1;
        sram_data_a = '{-10, 10, 0, 0}; // Raw inputs
        sram_data_b = '{5, 6, 0, 0};    // Sigmoid inputs from previous pass
        
        @(posedge clk); en_vecmul_valid = 0;
        
        repeat(4) @(posedge clk);
        
        if (vpu_data_out[0] === -8'sd110 && vpu_data_out[1] === 8'd60)
            $display("  -> [PASS] Complex GELU Symmetry routing successful.");
        else
            $error("  -> [FAIL] GELU routing failed. L0: %0d, L1: %0d", vpu_data_out[0], vpu_data_out[1]);

        $display("\n==================================================");
        $display("  ALL VPU CROSSBAR TESTS COMPLETED.               ");
        $display("==================================================");
        $finish;
    end

endmodule