

/*
 * @module   shared_lut_rom
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 */
module shared_lut_rom #(
    parameter int N = 8
)(
    input  logic [2:0] a_idx [N],  // Indices requested by the ALUs
    output logic [N-1:0] e_a   [N]   // Anchors returned to the ALUs
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
            e_a[i] = exp_lut[a_idx[i]];
        end
    end
endmodule