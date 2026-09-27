`timescale 1ns/1ps

/*
 * @module   ibert_divider
 * @brief    Unsigned integer restoring divider (classic shift/subtract/restore
 *           algorithm), WIDTH cycles of latency.
 * @details  Shared runtime-division primitive for ibert_softmax.sv (probability
 *           normalization) and ibert_rmsnorm.sv (Newton-Raphson sqrt iterations
 *           and the final x/rms divide). I-BERT (Kim et al., ICML 2021) calls
 *           for "an integer division" in both Algorithm 3 (softmax normalize)
 *           and Algorithm 4 (integer sqrt) but does not specify a hardware
 *           divider design -- this is a conventional restoring divider, a
 *           design choice made for this baseline, not part of I-BERT itself.
 *           Division by a compile-time CONSTANT (e.g. ibert_softmax.sv's
 *           z = floor(-x/ln2)) is a fundamentally cheaper operation (reduces
 *           to shifts/adds) and is done directly in the caller instead of
 *           routed through this divider.
 * @param    WIDTH   Operand/quotient/remainder width in bits.
 */
module ibert_divider #(
    parameter int WIDTH = 24
)(
    input  logic                clk,
    input  logic                rst_n,
    input  logic                valid_in,
    input  logic [WIDTH-1:0]    dividend,    // pre-shifted by the caller for fractional quotient bits
    input  logic [WIDTH-1:0]    divisor,
    output logic                busy,
    output logic                valid_out,
    output logic [WIDTH-1:0]    quotient,
    output logic [WIDTH-1:0]    remainder
);

    typedef enum logic [1:0] {IDLE, RUN, DONE} state_t;
    state_t state;

    logic [WIDTH-1:0]               a_reg;   // unconsumed dividend bits, MSB-first
    logic [WIDTH-1:0]               b_reg;
    logic [WIDTH-1:0]               q_reg;
    logic [WIDTH:0]                 r_reg;   // WIDTH+1 bits: guards the sign of the trial subtraction
    logic [$clog2(WIDTH+1)-1:0]     cnt;

    logic [WIDTH:0] r_shifted;
    logic [WIDTH:0] r_sub;

    always_comb begin
        r_shifted = {r_reg[WIDTH-1:0], a_reg[WIDTH-1]};
        r_sub     = r_shifted - {1'b0, b_reg};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            busy      <= 1'b0;
            valid_out <= 1'b0;
            a_reg <= '0; b_reg <= '0; q_reg <= '0; r_reg <= '0; cnt <= '0;
            quotient <= '0; remainder <= '0;
        end else begin
            valid_out <= 1'b0;
            case (state)
                IDLE: begin
                    if (valid_in) begin
                        if (divisor == '0) begin
                            // Divide-by-zero guard: saturate the quotient, no iteration needed.
                            quotient  <= '1;
                            remainder <= '0;
                            valid_out <= 1'b1;
                        end else begin
                            a_reg <= dividend;
                            b_reg <= divisor;
                            q_reg <= '0;
                            r_reg <= '0;
                            cnt   <= WIDTH[$clog2(WIDTH+1)-1:0];
                            busy  <= 1'b1;
                            state <= RUN;
                        end
                    end
                end
                RUN: begin
                    if (r_sub[WIDTH]) begin
                        // Trial subtraction went negative: restore, quotient bit = 0.
                        r_reg <= r_shifted;
                        q_reg <= {q_reg[WIDTH-2:0], 1'b0};
                    end else begin
                        r_reg <= r_sub;
                        q_reg <= {q_reg[WIDTH-2:0], 1'b1};
                    end
                    a_reg <= {a_reg[WIDTH-2:0], 1'b0};
                    cnt   <= cnt - 1'b1;
                    if (cnt == 1) state <= DONE;
                end
                DONE: begin
                    quotient  <= q_reg;
                    remainder <= r_reg[WIDTH-1:0];
                    valid_out <= 1'b1;
                    busy      <= 1'b0;
                    state     <= IDLE;
                end
            endcase
        end
    end

endmodule
