`timescale 1ns/1ps

// Reciprocal-sqrt unit fed the way the VPU feeds it: the rounded mean mk goes through the shared tr_ln_alu (same parameters as in
// tr_backbone_wrapper), whose leading-one position and mantissa bits index rsq_unit. Every mk the ln lane can receive is checked
// against tr_math_model.hpp::rms2_inv_mk (C++ mode "rsq" of verify_block.py).
module tr_rsq_unit_tb();

    localparam int FRAC_W   = 4;
    localparam int LN_EXTRA = 2;
    localparam int LN_IN_W  = 13;
    localparam int LN_OUT_W = FRAC_W + LN_EXTRA + 4;
    localparam int MSB_W    = $clog2(LN_IN_W);
    localparam int BITS     = FRAC_W + LN_EXTRA;
    localparam int INV_W    = 12;

    logic [LN_IN_W-1:0]          mk;
    logic signed [LN_OUT_W-1:0]  ln_y;
    logic [MSB_W-1:0]            msb;
    logic [BITS-1:0]             fk;
    logic [INV_W-1:0]            inv;

    tr_ln_alu #(.WIDTH(LN_IN_W), .BITS(BITS), .OUT_WIDTH(LN_OUT_W)) u_ln (.xq(mk), .yq(ln_y), .msb_o(msb), .fk_o(fk));
    rsq_unit  #(.MSB_W(MSB_W), .BITS(BITS), .INV_W(INV_W)) u_rsq (.msb(msb), .fk(fk), .inv(inv));

    int file_in, file_out, num_vecs, dummy, mk_raw;

    initial begin
        file_in  = $fopen("inputs.txt", "r");
        file_out = $fopen("hdl_out.txt", "w");
        if (!file_in || !file_out) begin
            $display("[ERROR] Could not open IO files! Ensure inputs.txt exists in workspace.");
            $finish;
        end
        dummy = $fscanf(file_in, "%0d\n", num_vecs);
        for (int i = 0; i < num_vecs; i++) begin
            dummy = $fscanf(file_in, "%d\n", mk_raw);
            mk = LN_IN_W'(mk_raw);
            #10;
            $fwrite(file_out, "%0d\n", inv);
        end
        $fclose(file_in);
        $fclose(file_out);
        $display("[RTL SIM] Successfully evaluated %0d RSQ vectors.", num_vecs);
        $finish;
    end
endmodule
