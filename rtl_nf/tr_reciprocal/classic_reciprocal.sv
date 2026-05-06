`timescale 1ns/1ps

module classic_reciprocal #(
    parameter int IN_WIDTH  = 16,
    parameter int OUT_WIDTH = 8,
    parameter int IN_FRAC   = 4,
    parameter int OUT_FRAC  = 4
)(
    input  logic                 clk,
    input  logic                 rst_n,
    input  logic                 start,
    input  logic [IN_WIDTH-1:0]  xq,
    
    output logic                 valid_out,
    output logic [OUT_WIDTH-1:0] yq    
);

    // Calculate fixed-point 1.0 
    // y_fixed = (2^(IN_FRAC + OUT_FRAC)) / x_fixed
    localparam int NUMERATOR_VAL = 1 << (IN_FRAC + OUT_FRAC);
    
    // Determine the divider width required to hold the static numerator
    localparam int DIV_WIDTH = (IN_WIDTH > (IN_FRAC + OUT_FRAC + 1)) ? 
                                IN_WIDTH : (IN_FRAC + OUT_FRAC + 1);

    logic [DIV_WIDTH-1:0] dividend;
    logic [DIV_WIDTH-1:0] divisor;
    logic [DIV_WIDTH-1:0] quotient;

    assign dividend = DIV_WIDTH'(NUMERATOR_VAL);
    assign divisor  = DIV_WIDTH'(xq);

    divu_int #(
        .WIDTH(DIV_WIDTH)
    ) u_divider (
        .clk   (clk),
        .rst_n (rst_n),
        .start (start),
        .busy  (),
        .done  (),
        .valid (valid_out),
        .dbz   (),
        .a     (dividend),
        .b     (divisor),
        .val   (quotient),
        .rem   ()
    );

    // Hardware saturation to prevent overflow when xq is extremely small
    always_comb begin
        if (quotient > ((1 << OUT_WIDTH) - 1)) begin
            yq = (1 << OUT_WIDTH) - 1; // Clamp to max value (e.g., 0xFF)
        end else begin
            yq = quotient[OUT_WIDTH-1:0];
        end
    end

endmodule