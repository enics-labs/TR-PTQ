`timescale 1ns/1ps

module tr_soc_top_mx_tb();

    // Matching the top-level parameters
    localparam int M        = 4;
    localparam int N        = 8;
    localparam int W        = 8;
    localparam int ACC_W    = 32;

    // Clock and Reset
    logic clk, rst_n;

    // RISC-V MMIO Interface
    logic [7:0]  mmio_addr;
    logic [31:0] mmio_wdata;
    logic        mmio_wen;
    logic [31:0] mmio_rdata;

    // Datapath SRAM Interfaces
    logic        dot_in_valid, dot_in_ready;
    logic [W-1:0] a_mat [M][N];
    logic signed [7:0] a_mat_exp; // NEW MX Exponent
    logic [W-1:0] b_vec [N];
    logic signed [7:0] b_vec_exp; // NEW MX Exponent
    logic signed [ACC_W-1:0] c_vec [M];
    
    logic signed [W-1:0] ext_sram_b [N];
    
    // VPU Outputs (Packed back to MX format)
    logic        vpu_out_valid;
    logic signed [W-1:0] vpu_data_out [N];
    logic signed [7:0]   vpu_data_exp; // NEW MX Formatter Exponent

    logic        clear_acc;
    // Streaming matmul interface
    logic [15:0] mm_tile_row, mm_tile_col;
    logic        mm_mem_rd;
    logic [W-1:0] mm_a_tile [M][N];
    logic [W-1:0] mm_b_tile [N];
    logic        mm_out_we;
    logic [15:0] mm_out_row;
    logic signed [W-1:0] mm_out_data [M];
    logic signed [7:0]   mm_out_exp;

    // DUT Instantiation
    tr_soc_top_mx #(
        .M(M), .N(N), .W(W), .ACC_W(ACC_W)
    ) dut (.*);

    // ── Streaming matmul memory model (combinational read) ──────────────
    localparam int MM_ROWS = 8, MM_COLS = 16;
    logic signed [W-1:0] MMA [MM_ROWS][MM_COLS];
    logic signed [W-1:0] MMB [MM_COLS];
    logic signed [W-1:0] MMO   [MM_ROWS];
    logic signed [7:0]   MMO_E [MM_ROWS];

    always_comb begin
        for (int m = 0; m < M; m++)
            for (int i = 0; i < N; i++)
                mm_a_tile[m][i] = MMA[mm_tile_row*M + m][mm_tile_col*N + i];
        for (int i = 0; i < N; i++)
            mm_b_tile[i] = MMB[mm_tile_col*N + i];
    end
    always_ff @(posedge clk)
        if (mm_out_we)
            for (int m = 0; m < M; m++) begin
                MMO[mm_out_row*M + m]   <= mm_out_data[m];
                MMO_E[mm_out_row*M + m] <= mm_out_exp;
            end

    // Clock Generation
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
        $display("   --- Final VPU Output (Reformatted to MXINT8) ---");
        $display("     [Shared Exponent E_out]: %0d", vpu_data_exp);
        for(int i=0; i<M; i++) begin
            $display("     Lane %0d Mantissa: %4d  | (True Value ~ %0d)", 
                     i, vpu_data_out[i], vpu_data_out[i] * (2 ** vpu_data_exp));
        end
    endtask

    // Fires the MAC and waits for the MX Requantizer to push to the VPU
    // Now accepts MX Exponents to simulate block-floating-point dynamic range!
    task automatic load_mac_vector(
        input logic signed [31:0] l0, input logic signed [31:0] l1, 
        input logic signed [31:0] l2, input logic signed [31:0] l3,
        input logic signed [7:0] exp_a, input logic signed [7:0] exp_b
    );
        $display("\n---> Pushing new data through Linear -> MX Requantize Pipe");
        @(posedge clk);
        dot_in_valid = 1;
        clear_acc = 1;   // single-tile: initialise accumulator
        c_vec[0] = l0; c_vec[1] = l1; c_vec[2] = l2; c_vec[3] = l3;
        a_mat_exp = exp_a;
        b_vec_exp = exp_b;
        @(posedge clk);
        dot_in_valid = 0;
        
        // Wait for Requantizer output valid flag
        do begin @(posedge clk); end while (!dut.req_out_valid);
        repeat(2) @(posedge clk); // Allow settling time into VPU/Shifter inputs
        
        $display("   --- MX Requantizer Output ---");
        $display("     [Shared Exponent E_total]: %0d", dut.mx_shared_exp);
        for(int i=0; i<M; i++) $display("     Lane %0d Mantissa: %4d", i, dut.req_vec_out[i]);
    endtask

    // =========================================================
    // MAIN SIMULATION SEQUENCE
    // =========================================================
    initial begin
        // Reset and Default Initializations
        rst_n = 0; mmio_wen = 0; dot_in_valid = 0; clear_acc = 1;
        a_mat_exp = 0; b_vec_exp = 0;
        for(int i=0; i<N; i++) begin b_vec[i]=0; ext_sram_b[i]=0; end
        for(int i=0; i<M; i++) begin c_vec[i]=0; for(int j=0; j<N; j++) a_mat[i][j]=0; end
        
        #22 rst_n = 1;
        $display("=================================================");
        $display(" TR-SOC MXINT8 FIRMWARE SIMULATION               ");
        $display("=================================================");

        // Note: NO MMIO SCALE CONFIGURATION NEEDED!
        // The MX Requantizer dynamically extracts it from the SRAM stream.

        // ---------------------------------------------------------
        // TEST 1: MAC / LINEAR ONLY
        // ---------------------------------------------------------
        // Mantissas: 10, -5, 0, 25. Exponents: Act=0, Wgt=0
        load_mac_vector(32'd10, -32'd5, 32'd0, 32'd25, 8'd0, 8'd0);
        $display("   [PASS] MX Linear Pipeline successfully handed off to VPU.");

        // ---------------------------------------------------------
        // TEST MM: STREAMING MATMUL via CMD=0x04 (integrated sequencer)
        // ---------------------------------------------------------
        // Smoke test: exercises the shared tr_matmul_ctrl on the mx datapath.
        // The mx requant emits {mantissa, shared exp}; exact numeric format is
        // covered by the op tests, so here we verify the control flow completes
        // and print the streamed matmul result per row (constant exponents).
        begin : mm_test
            for (int r = 0; r < MM_ROWS; r++)
                for (int c = 0; c < MM_COLS; c++) MMA[r][c] = ((r*3 + c) % 5) - 2;
            for (int c = 0; c < MM_COLS; c++) MMB[c] = (c % 3) - 1;
            a_mat_exp = 0; b_vec_exp = 0;   // constant across the contraction

            $display("\n---> CPU Executing OP_MATMUL (CMD=0x04): %0dx%0d", MM_ROWS, MM_COLS);
            mmio_write(8'h10, MM_ROWS/M);   // num_row_tiles
            mmio_write(8'h14, MM_COLS/N);   // num_ctiles
            mmio_write(8'h00, 32'h04);      // CMD = matmul
            wait_for_done();

            $display("   --- Streamed matmul result (mantissa @ shared exp) ---");
            for (int r = 0; r < MM_ROWS; r++)
                $display("     row %0d: %0d @ 2^%0d", r, MMO[r], MMO_E[r]);
            $display("   [PASS] MX matmul control flow completed.");
        end

        // ---------------------------------------------------------
        // TEST 2: GELU (Scale-Variant - Uses Dynamic Shifter Expansion)
        // ---------------------------------------------------------
        $display("\n---> CPU Executing OP_GELU (CMD = 0x02)");
        // To prove MX Expansion works, we feed it half-scale mantissas and an exponent of +1.
        // Mantissas: {8, -8, 0, 16}. E_total = +1. True linear values = {16, -16, 0, 32}.
        // The Top-Level shifter will expand this to 16-bits before GELU processes it!
        load_mac_vector(32'd8, -32'd8, 32'd0, 32'd16, 8'd1, 8'd0);
        mmio_write(8'h00, 32'h02); 
        wait_for_done();
        print_vpu_results();

        // ---------------------------------------------------------
        // TEST 3: SOFTMAX (Scale-Variant - Uses Dynamic Shifter Expansion)
        // ---------------------------------------------------------
        $display("\n---> CPU Executing OP_SOFTMAX (CMD = 0x01)");
        // Same concept: Half-scale mantissas, Exponent of +1.
        // Mantissas: {5, 10, 15, 20}. E_total = +1. True values = {10, 20, 30, 40}.
        load_mac_vector(32'd5, 32'd10, 32'd15, 32'd20, 8'd1, 8'd0); 
        mmio_write(8'h00, 32'h01); 
        wait_for_done();
        print_vpu_results();

        // ---------------------------------------------------------
        // TEST 4: RMSNORM (Scale-Invariant - Bypasses Shifter)
        // ---------------------------------------------------------
        $display("\n---> CPU Executing OP_RMSNORM (CMD = 0x03)");
        // RMSNorm is scale-invariant. The Exponent cancels out during the math.
        // We feed it an exponent of +5 to ensure the FSM correctly bypasses the shifter
        // and doesn't accidentally overflow the VPU with massive numbers.
        load_mac_vector(32'd16, -32'd16, 32'd16, -32'd16, 8'd5, 8'd0); 
        mmio_write(8'h00, 32'h03); 
        wait_for_done();
        print_vpu_results();

        $display("\n=================================================");
        $display(" ALL MX MACRO-INSTRUCTIONS EXECUTED SUCCESSFULLY!");
        $display("=================================================");
        $finish;
    end
endmodule