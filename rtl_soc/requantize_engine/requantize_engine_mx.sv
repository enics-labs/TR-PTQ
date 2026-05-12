`timescale 1ns/1ps

/*
 * @module   requantize_engine_mx
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 * @param    ACC_W           TODO: Add description
 * @param    OUT_W           TODO: Add description
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

    // -------------------------------------------------------------------------
    // 1. Calculate the Base MAC Exponent
    // -------------------------------------------------------------------------
    logic signed [7:0] base_exp;
    assign base_exp = exp_act_in + exp_weight_in;

    // -------------------------------------------------------------------------
    // 2. Find Maximum Absolute Value in the Block
    // -------------------------------------------------------------------------
    logic [ACC_W-1:0] abs_val [N];
    logic [ACC_W-1:0] max_abs;

    always_comb begin
        max_abs = '0;
        for (int i = 0; i < N; i++) begin
            // 2's complement absolute value
            abs_val[i] = (dot_in[i][ACC_W-1]) ? -dot_in[i] : dot_in[i];
            
            // Max Tree
            if (abs_val[i] > max_abs) begin
                max_abs = abs_val[i];
            end
        end
    end

    // -------------------------------------------------------------------------
    // 3. Count Leading Zeros to Determine Required Shift (S)
    // -------------------------------------------------------------------------
    logic [5:0] bit_width; // 6 bits to hold up to 32
    logic signed [OUT_W-1:0] shift_needed;

    always_comb begin
        bit_width = 0;
        for (int i = ACC_W-1; i >= 0; i--) begin
            if (max_abs[i] == 1'b1 && bit_width == 0) begin
                bit_width = i + 1;
            end
        end
        
        // Calculate how many bits we must shift right to fit into OUT_W (8 bits)
        // OUT_W - 1 is the magnitude portion (7 bits)
        if (bit_width > (OUT_W - 1)) begin
            shift_needed = bit_width - (OUT_W - 1);
        end else begin
            shift_needed = 0; // Already fits, no compression needed
        end
    end

    // -------------------------------------------------------------------------
    // 4. Shift, Update Exponent, and Register Outputs
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_out_valid <= 1'b0;
            exp_total_out <= '0;
            for (int i = 0; i < N; i++) req_vec_out[i] <= '0;
        end else begin
            req_out_valid <= dot_in_valid;
            if (dot_in_valid) begin
                // The new total exponent merges the MAC base scale and the compression shift
                exp_total_out <= base_exp + shift_needed;
                
                for (int i = 0; i < N; i++) begin
                    automatic logic signed [ACC_W-1:0] shifted_val;
                    
                    // Compress 32-bit down to 8-bit
                    shifted_val = dot_in[i] >>> shift_needed;
                    req_vec_out[i] <= shifted_val[OUT_W-1:0]; 
                end
            end
        end
    end

endmodule