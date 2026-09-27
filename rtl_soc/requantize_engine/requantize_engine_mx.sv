`timescale 1ns/1ps

/*
 * @module   requantize_engine_mx
 * @brief    2-stage pipelined MX-format requantizer: compresses N wide
 *           accumulator outputs down to OUT_W-bit mantissas plus a combined
 *           exponent.
 * @details  Stage 1 finds the max magnitude across the N dot_in lanes
 *           (combinational max-tree) and latches the base exponent
 *           (exp_act_in + exp_weight_in, the product of the two MX inputs'
 *           shared exponents that produced this accumulation). Stage 2
 *           leading-zero-counts the registered max to derive the minimum
 *           right-shift that fits it into OUT_W-1 magnitude bits, shifts
 *           every lane by that shared amount, and adds the shift onto the
 *           base exponent to produce exp_total_out -- the MX-format
 *           counterpart of requantize_engine_int's fixed-point (Mult>>>Shift)
 *           requantization.
 *
 * @param    N      Number of parallel lanes.
 * @param    ACC_W  Width of the incoming per-lane accumulator value (dot_in).
 * @param    OUT_W  Width of the outgoing per-lane mantissa (req_vec_out).
 */
module requantize_engine_mx #(
    parameter int N = 4,         
    parameter int ACC_W = 32,    
    parameter int OUT_W = 8      
)(
    input  logic clk,
    input  logic rst_n,
    input  logic dot_in_valid,
    output logic req_out_valid,
    input  logic signed [ACC_W-1:0] dot_in [N],
    input  logic signed [7:0]       exp_act_in,     
    input  logic signed [7:0]       exp_weight_in,  
    output logic signed [OUT_W-1:0] req_vec_out [N],
    output logic signed [7:0]       exp_total_out   
);

    // ========================================================================
    // PIPELINE STAGE 1: Max-Tree & Latch
    // ========================================================================
    logic signed [ACC_W-1:0] st1_dot_in [N];
    logic signed [7:0]       st1_base_exp;
    logic [ACC_W-1:0]        st1_max_abs;
    logic                    st1_valid;

    // Combinational Max-Tree
    logic [ACC_W-1:0] comb_abs_val [N];
    logic [ACC_W-1:0] comb_max_abs;

    always_comb begin
        comb_max_abs = '0;
        for (int i = 0; i < N; i++) begin
            comb_abs_val[i] = (dot_in[i][ACC_W-1]) ? -dot_in[i] : dot_in[i];
            if (comb_abs_val[i] > comb_max_abs) comb_max_abs = comb_abs_val[i];
        end
    end

    // Stage 1 Registers
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st1_valid <= 1'b0;
            st1_base_exp <= '0;
            st1_max_abs <= '0;
            for (int i = 0; i < N; i++) st1_dot_in[i] <= '0;
        end else begin
            st1_valid <= dot_in_valid;
            if (dot_in_valid) begin
                st1_base_exp <= exp_act_in + exp_weight_in;
                st1_max_abs  <= comb_max_abs;
                st1_dot_in   <= dot_in;
            end
        end
    end

    // ========================================================================
    // PIPELINE STAGE 2: LZC, Compression Shift, and Output
    // ========================================================================
    logic [5:0] bit_width;
    logic [7:0] comp_shift;

    // LZC on registered Max-Abs
    always_comb begin
        bit_width = 0;
        for (int i = ACC_W-1; i >= 0; i--) begin
            if (st1_max_abs[i] == 1'b1 && bit_width == 0) begin
                bit_width = i + 1;
            end
        end
        
        if (bit_width > (OUT_W - 1)) comp_shift = bit_width - (OUT_W - 1);
        else                         comp_shift = 0;
    end

    // Stage 2 Registers (Final Output)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_out_valid <= 1'b0;
            exp_total_out <= '0;
            for (int i = 0; i < N; i++) req_vec_out[i] <= '0;
        end else begin
            req_out_valid <= st1_valid;
            if (st1_valid) begin
                exp_total_out <= st1_base_exp + $signed({1'b0, comp_shift}); 
                
                for (int i = 0; i < N; i++) begin
                    automatic logic signed [ACC_W-1:0] shifted;
                    shifted = st1_dot_in[i] >>> comp_shift;
                    req_vec_out[i] <= shifted[OUT_W-1:0];
                end
            end
        end
    end

endmodule