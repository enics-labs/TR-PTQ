

/*
 * @module   shared_lut_rom
 * @brief    8-entry e^-k anchor table shared by every tr_exp_alu lane.
 * @details  Fixed Q0.8, format-independent ROM holding e^-1..e^-8 (the
 *           decay-only anchor points tr_exp_alu's round.sv rounds x to).
 *           Each of the N lanes independently indexes the same 8-entry
 *           table via its own a_idx (only a_idx's low 3 bits are used --
 *           the table itself does not grow with LUT_IDX_W).
 *
 * @param    N          Number of parallel lanes (independent index/lookup pairs).
 * @param    LUT_IDX_W  Width of each a_idx port; only bits [2:0] select the
 *                        (always 8-entry) table.
 */
module shared_lut_rom #(
    parameter int N = 8,
    parameter int LUT_IDX_W = 3  // table stays 8 entries regardless; only low 3 bits of a_idx are used
)(
    input  logic [LUT_IDX_W-1:0] a_idx [N],  // Indices requested by the ALUs
    output logic [7:0]           e_a   [N]   // Anchors returned to the ALUs (Q0.8, format-independent)
);
    // Definition of the Q4.4 LUT
    logic [7:0] exp_lut [0:7];

    assign exp_lut[0] = 8'd94;  // e^-1
    assign exp_lut[1] = 8'd35;  // e^-2
    assign exp_lut[2] = 8'd13;  // e^-3
    assign exp_lut[3] = 8'd5;   // e^-4
    assign exp_lut[4] = 8'd2;   // e^-5
    assign exp_lut[5] = 8'd1;   // e^-6
    assign exp_lut[6] = 8'd0;   // e^-7
    assign exp_lut[7] = 8'd0;   // e^-8

    // Combinational routing to all lanes
    always_comb begin
        for (int i = 0; i < N; i++) begin
            e_a[i] = exp_lut[a_idx[i][2:0]];
        end
    end
endmodule