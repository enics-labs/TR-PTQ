`timescale 1ns/1ps

// Tests tr_exp_alu (+ shared_lut_rom). WIDTH/FRAC_W are edited here per
// format before each run -- same convention as the synthesis top.
//
// Writes 5 columns per line: "x hw_final_y a_idx is_zero mantisa".
//   - hw_final_y (column index 1) replicates the VPU crossbar's final
//     downstream compose+scale (is_zero ? 255 : rom_e_a) * mantisa >>> FRAC_W
//     -- at the default FRAC_W=4 this is bit-identical to the original
//     hardcoded ">>4", so verify_block.py's "exp" regression (which only
//     ever reads column index 1) keeps working unchanged against the
//     existing Q4.4-only C++ golden model.
//   - a_idx/is_zero/mantisa are tr_exp_alu's own raw outputs, used by the
//     multi-format sweep (gen_exp_alu_golden.py / compare_exp_alu.py) to
//     verify round.sv/quadratic_divider.sv's generalization independent of
//     any downstream composition.
module tr_exp_alu_tb();

    localparam int WIDTH     = 8;
    localparam int FRAC_W    = 4;
    localparam int LUT_IDX_W = 3;
    localparam int ITER      = 2;
    localparam int N         = 8;  // N=8 so the ROM port matches [7:0] e_a [8]

    logic signed [WIDTH-1:0] x;
    logic [LUT_IDX_W-1:0]    a_idx;
    logic [WIDTH-1:0]        mantisa;
    logic                    is_zero;

    logic [7:0]              rom_e_a [N];
    logic [LUT_IDX_W-1:0]    rom_a_idx [N];

    always_comb begin
        rom_a_idx[0] = a_idx;
        for (int i = 1; i < N; i++) begin
            rom_a_idx[i] = '0;
        end
    end

    tr_exp_alu #(
        .WIDTH(WIDTH), .FRAC_W(FRAC_W), .LUT_IDX_W(LUT_IDX_W), .ITER(ITER)
    ) u_exp (
        .x(x), .a_idx(a_idx), .mantisa(mantisa), .is_zero(is_zero)
    );

    shared_lut_rom #(.N(N), .LUT_IDX_W(LUT_IDX_W)) u_rom (
        .a_idx(rom_a_idx), .e_a(rom_e_a)
    );

    int file_in, file_out, num_vecs, dummy;
    int stimulus_raw;
    logic [WIDTH-1:0] hw_final_y;

    initial begin
        file_in = $fopen("inputs.txt", "r");
        file_out = $fopen("hdl_out.txt", "w");

        if (!file_in || !file_out) begin
            $display("[ERROR] Could not open IO files! Ensure inputs.txt exists in workspace.");
            $finish;
        end

        dummy = $fscanf(file_in, "%d\n", num_vecs);

        for (int i = 0; i < num_vecs; i++) begin
            dummy = $fscanf(file_in, "%d\n", stimulus_raw);
            x = WIDTH'(stimulus_raw);

            #10;

            if (is_zero) begin
                hw_final_y = WIDTH'((32'(255) * 32'(mantisa)) >>> FRAC_W);
            end else begin
                hw_final_y = WIDTH'((32'(rom_e_a[0]) * 32'(mantisa)) >>> FRAC_W);
            end

            $fwrite(file_out, "%0d %0d %0d %0d %0d\n", $signed(x), hw_final_y, a_idx, is_zero, mantisa);
        end

        $fclose(file_in);
        $fclose(file_out);
        $display("[RTL SIM] Successfully evaluated %0d vectors at WIDTH=%0d FRAC_W=%0d.", num_vecs, WIDTH, FRAC_W);
        $finish;
    end
endmodule
