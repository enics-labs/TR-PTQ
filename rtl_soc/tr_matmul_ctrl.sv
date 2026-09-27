`timescale 1ns/1ps

/*
 * @module   tr_matmul_ctrl
 * @brief    Format-agnostic streaming-matmul sequencer (control only).
 * @details  Drives a shared linear datapath (dot_product_engine + requantizer)
 *           to compute OUT[r] = requant( SUM_c A[r][c]*B[c] ) over an
 *           (num_row_tiles*M) x (num_ctiles*N) matmul.
 *
 *           No datapath is instantiated here — this FSM only sequences it, so
 *           the SAME module is reused by tr_soc_top_int and tr_soc_top_mx (each
 *           wires it to its own int/mx dot+requant).
 *
 *           Per output-row-tile: stream num_ctiles tiles (clear_acc=1 on the
 *           first, =0 to accumulate), then capture the result on the
 *           num_ctiles-th requantizer valid — the final accumulated dot.  Using
 *           the valid COUNT (not a fixed drain latency) keeps it independent of
 *           the datapath's pipeline depth.
 *
 * @param    M  Output-tile height (rows per row-tile) of the driven dot_product_engine.
 * @param    N  Contraction-tile width (columns per c-tile) of the driven dot_product_engine.
 */
module tr_matmul_ctrl #(
    parameter int M = 4,
    parameter int N = 8
)(
    input  logic clk,
    input  logic rst_n,

    // Control
    input  logic         start,
    input  logic [15:0]  num_row_tiles,   // output rows / M
    input  logic [15:0]  num_ctiles,      // contraction cols / N
    output logic         busy,
    output logic         done,

    // Datapath drive (tile addresses + MAC control)
    output logic [15:0]  tile_row,        // current output-row-tile
    output logic [15:0]  tile_col,        // current contraction tile
    output logic         mem_rd,          // request: read tile (tile_row,tile_col)
    input  logic         mem_valid,       // response: a_tile/b_tile valid this cycle
    output logic         dot_in_valid,
    output logic         clear_acc,

    // Datapath feedback
    input  logic         req_valid,       // requantizer produced a result

    // Result capture strobe (grab the requant output for tile_row)
    output logic         result_we,
    output logic [15:0]  result_row
);

    typedef enum logic [1:0] {S_IDLE, S_RUN, S_DONE} state_t;
    state_t state;

    logic [15:0] rt, ct, vcount, n_rt, n_ct;

    logic feeding, consume, capture;
    // Request tile (rt,ct) while feeding; only consume it (feed the MAC and
    // advance) on mem_valid, so the sequencer tolerates any memory read latency.
    assign feeding      = (state == S_RUN) && (ct < n_ct);
    assign consume      = feeding && mem_valid;
    assign mem_rd       = feeding;
    assign dot_in_valid = consume;
    assign clear_acc    = consume && (ct == 16'd0);
    assign tile_row     = rt;
    assign tile_col     = ct;

    // The n_ct-th requant valid for this row-tile carries the full accumulated dot.
    assign capture      = (state == S_RUN) && req_valid && (vcount == n_ct - 16'd1);
    assign result_we    = capture;
    assign result_row   = rt;

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            rt <= '0; ct <= '0; vcount <= '0; n_rt <= '0; n_ct <= '0;
        end else begin
            case (state)
                S_IDLE: if (start) begin
                    n_rt   <= num_row_tiles;
                    n_ct   <= num_ctiles;
                    rt <= '0; ct <= '0; vcount <= '0;
                    state  <= S_RUN;
                end

                S_RUN: begin
                    if (consume)    ct     <= ct + 16'd1;
                    if (req_valid)  vcount <= vcount + 16'd1;
                    if (capture) begin
                        // capture wins over the vcount++ above (next row-tile)
                        if (rt == n_rt - 16'd1) begin
                            state <= S_DONE;
                        end else begin
                            rt     <= rt + 16'd1;
                            ct     <= '0;
                            vcount <= '0;
                        end
                    end
                end

                S_DONE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
