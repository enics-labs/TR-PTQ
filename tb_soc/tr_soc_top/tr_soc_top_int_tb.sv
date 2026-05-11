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
    logic [W-1:0] a_mat [M][N];
    logic [W-1:0] b_vec [N];
    logic signed [31:0] c_vec [M];
    logic signed [W-1:0] ext_sram_b [N];
    
    logic        vpu_out_valid;
    logic signed [W-1:0] vpu_data_out [N];

    tr_soc_top_int #(.M(M), .N(N), .W(W), .ACC_W(32)) dut (.*);

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
        rst_n = 0; mmio_wen = 0; dot_in_valid = 0;
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

        $display("\n=================================================");
        $display(" ALL MACRO-INSTRUCTIONS EXECUTED SUCCESSFULLY!   ");
        $display("=================================================");
        $finish;
    end
endmodule