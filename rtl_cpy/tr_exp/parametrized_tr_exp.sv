`timescale 1ns/1ps

module parameterized_tr_exp #(
    parameter int FRAC = 4,
    parameter int ITER = 2,
    parameter int EA_W = 8      // LUT Anchor Bit-Width
)(
    input  logic signed [7:0]  x,         // Q4
    output logic [EA_W-1:0]    e_a,       // Parameterized LUT Output
    output logic [7:0]         mantisa,   // Static 8-bit Mantissa
    output logic               is_zero
);

    // ---------------------------------------------------------
    // Extract integer and fractional parts
    // ---------------------------------------------------------
    logic              is_ceil;   
    logic [2:0]        a_idx;   
    logic [3:0]        x_frac;  

    q4_4_round_neg uut (
        .x(x),
        .is_zero(is_zero),
        .is_ceil(is_ceil),
        .fliped_rounded_int(a_idx)
    );

    assign x_frac = x[3:0];

    // ---------------------------------------------------------
    // Stage 0 : Parameterized LUT read & e^0 bypass
    // ---------------------------------------------------------
    generate
        if (EA_W == 16) begin : gen_lut_16
            // 16-BIT MODE: Scale by 65536
            logic [15:0] exp_lut_16 [0:7];
            assign exp_lut_16[0] = 16'd24109; // e^-1
            assign exp_lut_16[1] = 16'd8869;  // e^-2
            assign exp_lut_16[2] = 16'd3263;  // e^-3
            assign exp_lut_16[3] = 16'd1200;  // e^-4
            assign exp_lut_16[4] = 16'd442;   // e^-5
            assign exp_lut_16[5] = 16'd162;   // e^-6
            assign exp_lut_16[6] = 16'd60;    // e^-7
            assign exp_lut_16[7] = 16'd22;    // e^-8
            assign e_a = is_zero ? 16'd65535 : exp_lut_16[a_idx]; // e^0
            
        end else if (EA_W == 4) begin : gen_lut_4
            // 4-BIT MODE: Scale by 16
            logic [3:0] exp_lut_4 [0:7];
            assign exp_lut_4[0] = 4'd6;       // e^-1
            assign exp_lut_4[1] = 4'd2;       // e^-2
            assign exp_lut_4[2] = 4'd1;       // e^-3
            assign exp_lut_4[3] = 4'd0;       // e^-4 (Flushes to 0 below here)
            assign exp_lut_4[4] = 4'd0;       // e^-5
            assign exp_lut_4[5] = 4'd0;       // e^-6
            assign exp_lut_4[6] = 4'd0;       // e^-7
            assign exp_lut_4[7] = 4'd0;       // e^-8
            assign e_a = is_zero ? 4'd15 : exp_lut_4[a_idx];      // e^0

        end else begin : gen_lut_8
            // 8-BIT MODE: Scale by 256 (Default)
            logic [7:0] exp_lut_8 [0:7];
            assign exp_lut_8[0] = 8'd94;      // e^-1
            assign exp_lut_8[1] = 8'd35;      // e^-2
            assign exp_lut_8[2] = 8'd13;      // e^-3
            assign exp_lut_8[3] = 8'd5;       // e^-4
            assign exp_lut_8[4] = 8'd2;       // e^-5
            assign exp_lut_8[5] = 8'd1;       // e^-6
            assign exp_lut_8[6] = 8'd0;       // e^-7
            assign exp_lut_8[7] = 8'd0;       // e^-8
            assign e_a = is_zero ? 8'd255 : exp_lut_8[a_idx];     // e^0
        end
    endgenerate

    // ---------------------------------------------------------
    // Stage 1 : first-order term 1+x-a
    // ---------------------------------------------------------
    logic [4:0] first_order;
    assign first_order = {~is_ceil, x_frac};

    // ---------------------------------------------------------
    // Stage 2 : second-order term (x-a)^2/2
    // ---------------------------------------------------------
    logic [1:0] xa_square;
    logic [5:0] second_order;
    
    q4_4_quadratic_divider qd(
        .x(x_frac),      
        .y(xa_square)    
    );

    assign second_order = {1'b0, first_order} + {4'b0, xa_square};

    // ---------------------------------------------------------
    // Stage 3 : final computation
    // ---------------------------------------------------------
    always_comb begin
        case (ITER)
            0: mantisa = 8'd0; 
            1: mantisa = {3'b000, first_order}; 
            2: mantisa = {2'b00, second_order}; 
            default: mantisa = 8'd0; 
        endcase
    end

endmodule