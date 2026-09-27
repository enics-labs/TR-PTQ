/*
 * @module   piped_max
 * @brief    Fully-pipelined binary-tree max reduction over NUM_INPUTS lanes.
 * @details  Builds log2(NUM_INPUTS) comparison levels, each registered (one
 *           pipeline stage per tree level), so max_out emerges
 *           STAGES=$clog2(NUM_INPUTS) cycles after valid_in, with valid_out
 *           tracking it through a matching shift-register delay. Used where
 *           a single-cycle combinational max-scan (as in a plain for-loop)
 *           would be too deep a critical path for a wide reduction.
 *
 * @param    NUM_INPUTS  Number of parallel input lanes to reduce.
 * @param    DATA_WIDTH  Signed data width of in_data/max_out.
 */
module piped_max #(
    parameter int NUM_INPUTS = 8,  
    parameter int DATA_WIDTH = 8  
)(
    input  logic clk,
    input  logic rst_n,
    input  logic valid_in,
    input  logic signed [DATA_WIDTH-1:0]  in_data [NUM_INPUTS],
    output logic signed [DATA_WIDTH-1:0]  max_out, 
    output logic valid_out
);

    localparam int STAGES = $clog2(NUM_INPUTS);

    // --- 1. Explicit Declaration of the Tree Structure ---
    // We create an array of logic arrays. 
    // Each 'row' represents a stage of the pipeline.
    genvar s, i;
    generate
        for (s = 0; s <= STAGES; s++) begin : stage_decl
            localparam int WIDTH = NUM_INPUTS >> s;
            logic signed [DATA_WIDTH-1:0] data [WIDTH];
        end
    endgenerate

    // --- 2. Data Logic ---
    generate
        // Connect inputs to the first stage
        for (i = 0; i < NUM_INPUTS; i++) begin : input_bind
            assign stage_decl[0].data[i] = in_data[i];
        end

        // Build the comparison tree
        for (s = 0; s < STAGES; s++) begin : tree_level
            localparam int NEXT_WIDTH = NUM_INPUTS >> (s + 1);
            
            for (i = 0; i < NEXT_WIDTH; i++) begin : comp_block
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        stage_decl[s+1].data[i] <= '0;
                    end else begin
                        // The actual hardware comparison
                        if (stage_decl[s].data[2*i] >= stage_decl[s].data[2*i+1])
                            stage_decl[s+1].data[i] <= stage_decl[s].data[2*i];
                        else
                            stage_decl[s+1].data[i] <= stage_decl[s].data[2*i+1];
                    end
                end
            end
        end
    endgenerate

    // --- 3. Valid Signal Pipeline (Shift Register) ---
    logic [STAGES-1:0] v_pipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_pipe <= '0;
        end else begin
            // v_pipe[0] is the current input,
            // others are delayed versions
            v_pipe <= {v_pipe[STAGES-2:0], valid_in};
        end
    end

    // --- 4. Final Assignments ---
    assign max_out   = stage_decl[STAGES].data[0];
    assign valid_out = v_pipe[STAGES-1];

endmodule