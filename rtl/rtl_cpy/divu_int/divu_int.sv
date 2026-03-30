
/* DO NOT TOUCH!!! */

module divu_int #(parameter WIDTH=16) ( // width of numbers in bits
    input wire clk,              // clock
    input wire rst_n,              // reset
    input wire start,            // start calculation
    output reg    busy,             // calculation in progress
    output reg    done,             // calculation is complete (high for one tick)
    output reg    valid,            // result is valid
    output reg    dbz,              // divide by zero
    input wire [WIDTH-1:0] a,    // dividend (numerator)
    input wire [WIDTH-1:0] b,    // divisor (denominator)
    output     reg [WIDTH-1:0] val,  // result value: quotient
    output     reg [WIDTH-1:0] rem   // result: remainder
    );

    reg [WIDTH-1:0] b1;             // copy of divisor
    reg [WIDTH-1:0] quo, quo_next;  // intermediate quotient
    reg [WIDTH:0] acc, acc_next;    // accumulator (1 bit wider)
    reg [$clog2(WIDTH)-1:0] i;      // iteration counter

    // division algorithm iteration
    always_comb begin
        if (acc >= {1'b0, b1}) begin
            acc_next = acc - b1;
            {acc_next, quo_next} = {acc_next[WIDTH-1:0], quo, 1'b1};  // If positive, quotient gets a 1.  
        end else begin
            {acc_next, quo_next} = {acc, quo} << 1;
        end
    end

    // calculation control
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 0;
            done <= 0;
            valid <= 0;
            dbz <= 0;
            val <= 0;
            rem <= 0;
        end else begin
            done <= 0;
            if (start) begin
                valid <= 0;
                i <= 0;
                if (b == 0) begin  // catch divide by zero
                    busy <= 0;
                    done <= 1;
                    dbz <= 1;
                end else begin
                    busy <= 1;
                    dbz <= 0;
                    b1 <= b;
                    {acc, quo} <= {{WIDTH{1'b0}}, a, 1'b0};  // initialize 
                end
            end else if (busy) begin
                if (i == WIDTH-1) begin  // we're done
                    busy <= 0;
                    done <= 1;
                    valid <= 1;
                    val <= quo_next;
                    rem <= acc_next[WIDTH:1];  // undo final shift
                end else begin  // next iteration
                    i <= i + 1;
                    acc <= acc_next;
                    quo <= quo_next;
                end
            end
        end
    end
endmodule