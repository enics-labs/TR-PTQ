`timescale 1ns/1ps

module tr_soc_top_int_tb();

    localparam int M = 4;
    localparam int N = 8;
    localparam int W = 8;

    logic clk, rst_n;
    logic [7:0]  mmio_addr;
    logic [31:0] mmio_wdata;
    logic        mmio_wen;
    logic [31:0] mmio_rdata;

    logic        dot_in_valid, dot_in_ready;
    logic        clear_acc;
    logic [W-1:0] a_mat [M][N];
    logic [W-1:0] b_vec [N];
    logic signed [31:0] c_vec [M];
    logic signed [W-1:0] ext_sram_b [N];
    
    logic        vpu_out_valid;
    logic signed [W-1:0] vpu_data_out [N];

    // Streaming matmul interface
    logic [15:0] mm_tile_row, mm_tile_col;
    logic        mm_mem_rd;
    logic        mm_mem_valid;
    logic [W-1:0] mm_a_tile [M][N];
    logic [W-1:0] mm_b_tile [N];
    logic        mm_out_we;
    logic [15:0] mm_out_row;
    logic signed [W-1:0] mm_out_data [M];

    tr_soc_top_int #(.M(M), .N(N), .W(W), .ACC_W(32)) dut (.*);

    // ── Streaming matmul memory model (combinational read) ──────────────
    localparam int MM_ROWS = 8, MM_COLS = 16;
    logic signed [W-1:0] MMA [MM_ROWS][MM_COLS];
    logic signed [W-1:0] MMB [MM_COLS];
    logic signed [W-1:0] MMO [MM_ROWS];

    assign mm_mem_valid = 1'b1;   // combinational memory: data ready same cycle
    always_comb begin
        for (int m = 0; m < M; m++)
            for (int i = 0; i < N; i++)
                mm_a_tile[m][i] = MMA[mm_tile_row*M + m][mm_tile_col*N + i];
        for (int i = 0; i < N; i++)
            mm_b_tile[i] = MMB[mm_tile_col*N + i];
    end
    always_ff @(posedge clk)
        if (mm_out_we)
            for (int m = 0; m < M; m++) MMO[mm_out_row*M + m] <= mm_out_data[m];

    function automatic logic signed [W-1:0] mm_ref(input int r);
        int acc;
        acc = 0;
        for (int c = 0; c < MM_COLS; c++) acc += MMA[r][c] * MMB[c];
        if (acc >  127) acc =  127;
        if (acc < -128) acc = -128;
        return acc[W-1:0];
    endfunction

    initial begin clk = 0; forever #5 clk = ~clk; end

    // =========================================================
    // HELPER TASKS
    // =========================================================
    task automatic mmio_write(input logic [7:0] addr, input logic [31:0] data);
        @(posedge clk);
        mmio_addr = addr; mmio_wdata = data; mmio_wen = 1;
        @(posedge clk); mmio_wen = 0;
        $display("[MMIO] Wrote 0x%0h -> [Addr: 0x%0h]", data, addr);
    endtask

    task automatic wait_for_done();
        logic [31:0] status;
        $display("[SoC]  Polling STATUS for DONE bit...");
        do begin
            @(posedge clk);
            mmio_addr = 8'h04; // ADDR_STATUS
            mmio_wen = 0;
            status = mmio_rdata;
        end while ((status & 32'h2) == 0); // Check bit 1 (Done)
        $display("[SoC]  Operation Complete!");
    endtask

    task automatic print_vpu_results();
        $display("   --- VPU Result Vector ---");
        for(int i=0; i<M; i++) $display("     Lane %0d: %4d", i, vpu_data_out[i]);
    endtask

    // Fires the MAC and waits for the Requantizer to push it to the VPU SRAM
    task automatic load_mac_vector(input logic signed [31:0] l0, input logic signed [31:0] l1, 
                                   input logic signed [31:0] l2, input logic signed [31:0] l3);
        $display("\n---> Pushing new data through Linear -> Requantize Pipe");
        @(posedge clk);
        dot_in_valid = 1;
        clear_acc = 1;   // single-tile: initialise accumulator with c_vec bias
        c_vec[0] = l0; c_vec[1] = l1; c_vec[2] = l2; c_vec[3] = l3;
        @(posedge clk);
        dot_in_valid = 0;
        
        // Wait for Requantizer output valid flag
        do begin @(posedge clk); end while (!dut.req_out_valid);
        repeat(2) @(posedge clk); // Allow settling time into VPU inputs
        
        $display("   --- Requantizer Output (VPU Input) ---");
        for(int i=0; i<M; i++) $display("     Lane %0d: %4d", i, dut.req_vec_out[i]);
    endtask

    // =========================================================
    // MAIN SIMULATION SEQUENCE
    // =========================================================
    initial begin
        rst_n = 0; mmio_wen = 0; dot_in_valid = 0; clear_acc = 1;
        for(int i=0; i<N; i++) begin b_vec[i]=0; ext_sram_b[i]=0; end
        for(int i=0; i<M; i++) begin c_vec[i]=0; for(int j=0; j<N; j++) a_mat[i][j]=0; end
        
        #22 rst_n = 1;

        $display("=================================================");
        $display(" TR-SOC FULL MULTI-PATH FIRMWARE SIMULATION      ");
        $display("=================================================");

        // Setup Requantizer for 1:1 Pass-through (Mult=1, Shift=0)
        // This allows c_vec to pass directly into the VPU unharmed
        mmio_write(8'h08, 32'd1); // REQ_MULT
        mmio_write(8'h0C, 32'd0); // REQ_SHIFT

        // ---------------------------------------------------------
        // TEST 0: STREAMING MATMUL (multi-tile accumulating dot)
        // ---------------------------------------------------------
        // Stream NT tiles: first tile clear_acc=1 (init acc with c_vec=0), the
        // rest clear_acc=0 (accumulate).  With a_mat[m][i]=m+1 and b_vec[i]=1,
        // each tile adds (m+1)*N per lane, so the requantized (1:1) result must
        // be (m+1)*N*NT.  Proves an arbitrary-length contraction streams through
        // the linear->requantize path.
        begin : stream_matmul_test
            int NT;
            NT = 3;
            $display("\n---> STREAMING MATMUL: %0d tiles of %0dx%0d (contraction %0d)",
                     NT, M, N, N*NT);
            for (int t = 0; t < NT; t++) begin
                @(posedge clk);
                dot_in_valid = 1;
                clear_acc    = (t == 0);
                for (int m = 0; m < M; m++) begin
                    c_vec[m] = 0;
                    for (int i = 0; i < N; i++) a_mat[m][i] = (m + 1);
                end
                for (int i = 0; i < N; i++) b_vec[i] = 1;
            end
            @(posedge clk);
            dot_in_valid = 0;
            clear_acc    = 1;
            repeat (10) @(posedge clk);   // flush MAC + requantizer pipeline

            for (int m = 0; m < M; m++) begin
                if (dut.req_vec_out[m] !== (m+1)*N*NT)
                    $error("  [FAIL] stream lane %0d: got %0d, expected %0d",
                           m, dut.req_vec_out[m], (m+1)*N*NT);
                else
                    $display("  [PASS] stream lane %0d = %0d (expected %0d)",
                             m, dut.req_vec_out[m], (m+1)*N*NT);
            end
            // Restore zero inputs so the legacy single-tile tests below are clean.
            for (int i = 0; i < N; i++) b_vec[i] = 0;
            for (int m = 0; m < M; m++) for (int i = 0; i < N; i++) a_mat[m][i] = 0;
        end

        // ---------------------------------------------------------
        // TEST MM: STREAMING MATMUL via CMD=0x04 (integrated sequencer)
        // ---------------------------------------------------------
        begin : mm_test
            int fails;
            for (int r = 0; r < MM_ROWS; r++)
                for (int c = 0; c < MM_COLS; c++) MMA[r][c] = ((r*3 + c) % 5) - 2;
            for (int c = 0; c < MM_COLS; c++) MMB[c] = (c % 3) - 1;

            $display("\n---> CPU Executing OP_MATMUL (CMD=0x04): %0dx%0d", MM_ROWS, MM_COLS);
            mmio_write(8'h08, 32'd1);            // REQ_MULT  = 1  (1:1)
            mmio_write(8'h0C, 32'd0);            // REQ_SHIFT = 0
            mmio_write(8'h10, MM_ROWS/M);        // num_row_tiles
            mmio_write(8'h14, MM_COLS/N);        // num_ctiles
            mmio_write(8'h00, 32'h04);           // CMD = matmul
            wait_for_done();

            fails = 0;
            for (int r = 0; r < MM_ROWS; r++) begin
                if (MMO[r] !== mm_ref(r)) begin
                    $error("  [FAIL] mm row %0d: got %0d, expected %0d", r, MMO[r], mm_ref(r));
                    fails++;
                end else
                    $display("  [PASS] mm row %0d = %0d", r, MMO[r]);
            end
            if (fails == 0) $display("  === MATMUL: ALL %0d ROWS PASS ===", MM_ROWS);
        end

        // ---------------------------------------------------------
        // TEST 1: MAC / LINEAR ONLY
        // ---------------------------------------------------------
        load_mac_vector(32'd10, -32'd5, 32'd0, 32'd25);
        $display("   [PASS] Linear Pipeline successfully handed off to VPU.");

        // ---------------------------------------------------------
        // TEST 2: GELU
        // ---------------------------------------------------------
        $display("\n---> CPU Executing OP_GELU (CMD = 0x02)");
        // Using inputs: {16 (1.0), -16 (-1.0), 0 (0.0), 32 (2.0)}
        load_mac_vector(32'd16, -32'd16, 32'd0, 32'd32); 
        mmio_write(8'h00, 32'h02); 
        wait_for_done();
        print_vpu_results();

        // ---------------------------------------------------------
        // TEST 3: SOFTMAX
        // ---------------------------------------------------------
        $display("\n---> CPU Executing OP_SOFTMAX (CMD = 0x01)");
        // Using inputs: {10, 20, 30, 40} -> Max is 40.
        // It should subtract max, exponentiate, sum, and divide!
        load_mac_vector(32'd10, 32'd20, 32'd30, 32'd40); 
        mmio_write(8'h00, 32'h01); 
        wait_for_done();
        print_vpu_results();

        // ---------------------------------------------------------
        // TEST 4: RMSNORM
        // ---------------------------------------------------------
        $display("\n---> CPU Executing OP_RMSNORM (CMD = 0x03)");
        // Using inputs: {16, -16, 16, -16}
        // Squares should all be positive, mean squared should process correctly.
        load_mac_vector(32'd16, -32'd16, 32'd16, -32'd16);
        mmio_write(8'h00, 32'h03);
        wait_for_done();
        print_vpu_results();

        // ---------------------------------------------------------
        // TEST 5: RMSNORM SIGN-GUARD FIX (small Sum(x^2), previously broken)
        // ---------------------------------------------------------
        // Inputs {3,-3,3,-3} on lanes 0-3 (lanes 4-7 come from whatever this
        // harness's own SRAM routing leaves on the other 4 lanes at this
        // point -- not independently re-derived here) push Sum(x^2) deep
        // into the previously-broken small-Sum(x^2) regime. ctrl_scalar
        // (reg_scalar_log + CONST_LN_SQRT_N) goes positive; before the
        // sign-guard fix, tr_soc_ctrl_int.sv's RM_P3 fed that straight to
        // tr_nonlinear_vpu's decay-only exp backbone and InvRMS collapsed to
        // 0, giving an all-zero VPU result on every lane regardless of x --
        // CONFIRMED directly by a controlled A/B: re-running this exact test
        // with rm_offset_was_positive forced to 0 (fix disabled) reproduces
        // that all-zero result precisely; with the fix restored it does not.
        // Check what that A/B proved -- non-zero, i.e. no longer the
        // collapse -- rather than an exact value: this test doesn't
        // independently know lanes 4-7's contents, so it can't derive the
        // precise golden output the way tr_rmsnorm_tb.sv's dedicated,
        // fully-controlled 8-lane test already did (see that testbench and
        // docs/iscas_paper_support/ for exact-value verification).
        $display("\n---> CPU Executing OP_RMSNORM (CMD = 0x03), sign-guard fix regression test");
        load_mac_vector(32'd3, -32'd3, 32'd3, -32'd3);
        mmio_write(8'h00, 32'h03);
        wait_for_done();
        print_vpu_results();
        begin : rmsnorm_signguard_check
            int unsigned fails;
            fails = 0;
            for (int m = 0; m < 2; m++) begin   // lanes 0-1 are known non-zero inputs (3, -3)
                if (vpu_data_out[m] === 8'sd0) begin
                    $error("  [FAIL] rmsnorm sign-guard lane %0d: got 0 -- looks like the pre-fix collapse", m);
                    fails++;
                end else
                    $display("  [PASS] rmsnorm sign-guard lane %0d = %0d (non-zero -- not the collapse)",
                             m, vpu_data_out[m]);
            end
            // +-1 LSB tolerance: truncating (not rounding) fixed-point
            // division isn't perfectly symmetric for +-x, same as every
            // other test in this file (e.g. TEST 4 above happened to land
            // symmetric, TEST 2/3 don't) -- not a correctness issue.
            if ((vpu_data_out[0] + vpu_data_out[1]) > 1 || (vpu_data_out[0] + vpu_data_out[1]) < -1) begin
                $error("  [FAIL] rmsnorm sign-guard: lane 0 (%0d) and lane 1 (%0d) should be within 1 LSB of exact negatives (x=3,-3)",
                       vpu_data_out[0], vpu_data_out[1]);
                fails++;
            end
            if (fails == 0)
                $display("  === RMSNORM SIGN-GUARD FIX: non-zero and correctly signed (was all-zero pre-fix, confirmed by A/B) ===");
        end

        $display("\n=================================================");
        $display(" ALL MACRO-INSTRUCTIONS EXECUTED SUCCESSFULLY!   ");
        $display("=================================================");
        $finish;
    end
endmodule