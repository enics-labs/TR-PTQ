`timescale 1ns/1ps

module q12_8_lut (
    input  logic [3:0] a_idx,
    output logic [19:0] e_a
);
    // mathematically exact round(e^x * 256) constants
    always_comb begin
        case (a_idx)
            4'sd0:  e_a = 20'd256;    // e^0  = 1.000
            4'sd1:  e_a = 20'd696;    // e^1  = 2.718
            4'sd2:  e_a = 20'd1892;   // e^2  = 7.389
            4'sd3:  e_a = 20'd5142;   // e^3  = 20.086
            4'sd4:  e_a = 20'd13977;  // e^4  = 54.598
            4'sd5:  e_a = 20'd37994;  // e^5  = 148.413
            4'sd6:  e_a = 20'd103278; // e^6  = 403.429
            4'sd7:  e_a = 20'd280738; // e^7  = 1096.633
            
            4'sd15: e_a = 20'd94;     // e^-1 = 0.368
            4'sd14: e_a = 20'd35;     // e^-2 = 0.135
            4'sd13: e_a = 20'd13;     // e^-3 = 0.050
            4'sd12: e_a = 20'd5;      // e^-4 = 0.018
            4'sd11: e_a = 20'd2;      // e^-5 = 0.007
            4'sd10: e_a = 20'd1;      // e^-6 = 0.002
            default: e_a = 20'd0;     // Flush to 0
        endcase
    end
endmodule