`timescale 1ns/1ps

module formatter_mx #(
    parameter int N = 4,
    parameter int VPU_W = 16, // Width of data coming from VPU
    parameter int MX_W = 8    // Width of target mantissa
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 valid_in,
    input  logic signed [VPU_W-1:0] vpu_data_in [N],
    
    output logic                 valid_out,
    output logic signed [MX_W-1:0]  mx_mantissas [N],
    output logic signed [7:0]    mx_shared_exp
);

    // -------------------------------------------------------------------------
    // 1. Calculate Absolute Values and Find Maximum
    // -------------------------------------------------------------------------
    logic [VPU_W-1:0] abs_val [N];
    logic [VPU_W-1:0] max_abs;

    always_comb begin
        max_abs = '0;
        for (int i = 0; i < N; i++) begin
            // 2's complement absolute value
            abs_val[i] = (vpu_data_in[i][VPU_W-1]) ? -vpu_data_in[i] : vpu_data_in[i];
            
            // Combinational Max Tree
            if (abs_val[i] > max_abs) begin
                max_abs = abs_val[i];
            end
        end
    end

    // -------------------------------------------------------------------------
    // 2. Count Leading Zeros (CLZ) -> Bit Width -> E_out
    // -------------------------------------------------------------------------
    logic [4:0] bit_width;
    logic signed [7:0] calculated_exp;

    // Simple priority encoder to find active bit width
    always_comb begin
        bit_width = 0;
        for (int i = VPU_W-1; i >= 0; i--) begin
            if (max_abs[i] == 1'b1 && bit_width == 0) begin
                bit_width = i + 1; // e.g., if bit 10 is the highest '1', width is 11
            end
        end
        
        // MX_W - 1 is the magnitude bits of our target format (e.g., 7 bits for INT8)
        // Required Shift = Bit Width - Target Magnitude Width
        if (bit_width > (MX_W - 1)) begin
            calculated_exp = bit_width - (MX_W - 1);
        end else begin
            calculated_exp = 0; // It already fits in 8 bits, no shift needed
        end
    end

    // -------------------------------------------------------------------------
    // 3. Shift and Register Outputs
    // -------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            mx_shared_exp <= '0;
            for (int i = 0; i < N; i++) mx_mantissas[i] <= '0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                mx_shared_exp <= calculated_exp;
                for (int i = 0; i < N; i++) begin
                    // Arithmetic Right Shift to compress back to 8-bit
                    // Using rounding-to-zero (truncation) for simplicity and area
                    automatic logic signed [VPU_W-1:0] shifted;
                    shifted = vpu_data_in[i] >>> calculated_exp;
                    mx_mantissas[i] <= shifted[MX_W-1:0]; 
                end
            end
        end
    end

endmodule