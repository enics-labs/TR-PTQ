`timescale 1ns/1ps

module tr_exp_tb();

    // Hardware Parameters
    localparam int WIDTH     = 8;
    localparam int FRAC_W    = 4;
    localparam int LUT_IDX_W = 3;
    localparam int ITER      = 2; // Second-order Taylor
    
    // Set N=8 so the ROM port expands to [7:0] e_a [8]
    localparam int N         = 8; 

    // Interconnects
    logic signed [WIDTH-1:0] x;
    logic [LUT_IDX_W-1:0]    a_idx;
    logic [WIDTH-1:0]        mantisa;
    logic                    is_zero;
    
    // Arrays sized to match N=8
    logic [7:0]              rom_e_a [N];
    logic [LUT_IDX_W-1:0]    rom_a_idx [N];

    // Tie lane 0 to our unit under test, zero out the rest
    always_comb begin
        rom_a_idx[0] = a_idx;
        for (int i = 1; i < N; i++) begin
            rom_a_idx[i] = '0;
        end
    end

    // Instantiate Unit Under Test
    tr_exp_alu #(
        .WIDTH(WIDTH), .FRAC_W(FRAC_W), .LUT_IDX_W(LUT_IDX_W), .ITER(ITER)
    ) u_exp (
        .x(x), .a_idx(a_idx), .mantisa(mantisa), .is_zero(is_zero)
    );

    // Instantiate ROM with N=8 to force [7:0] width matching
    shared_lut_rom #(.N(N)) u_rom (
        .a_idx(rom_a_idx), .e_a(rom_e_a)
    );

    int file_in, file_out, num_vecs, dummy;
    int stimulus_raw;
    logic [7:0] hw_final_y;

    initial begin
        file_in = $fopen("inputs.txt", "r");
        file_out = $fopen("hdl_out.txt", "w");
        
        if (!file_in || !file_out) begin
            $display("[ERROR] Could not open IO files! Ensure inputs.txt exists in workspace.");
            $finish;
        end

        // Read total vector count
        dummy = $fscanf(file_in, "%d\n", num_vecs);

        for (int i = 0; i < num_vecs; i++) begin
            dummy = $fscanf(file_in, "%d\n", stimulus_raw);
            x = WIDTH'(stimulus_raw);
            
            #10; // Wait for combinational logic to settle

            // Replicate the final VecMul scale-down in the VPU crossbar
            if (is_zero) begin
                hw_final_y = (16'(255) * 16'(mantisa)) >> 4;
            end else begin
                // Standard ROM lookup for e^-1 through e^-8
                hw_final_y = (16'(rom_e_a[0]) * 16'(mantisa)) >> 4;
            end
            
            // Format output exactly like the C++ expected log
            $fwrite(file_out, "%0d %0d\n", $signed(x), hw_final_y);
        end

        $fclose(file_in);
        $fclose(file_out);
        $display("[RTL SIM] Successfully evaluated %0d vectors.", num_vecs);
        $finish;
    end
endmodule