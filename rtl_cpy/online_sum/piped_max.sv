`timescale 1ns/1ps

module piped_max #(
    parameter int NUM_INPUTS = 8,  
    parameter int DATA_WIDTH = 8,
    parameter bit IS_SIGNED  = 1   // 1: Signed comparison, 0: Unsigned comparison
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          valid_in,
    input  logic signed [DATA_WIDTH-1:0]  in_data [NUM_INPUTS],
    output logic signed [DATA_WIDTH-1:0]  max_out, 
    output logic                          valid_out
);

    localparam int STAGES = $clog2(NUM_INPUTS);

    // --- 1. Tree Structure Declaration ---
    generate
        for (genvar s = 0; s <= STAGES; s++) begin : stage_decl
            localparam int WIDTH = NUM_INPUTS >> s;
            logic signed [DATA_WIDTH-1:0] data [WIDTH];
        end
    endgenerate

    // --- 2. Data Logic & Comparisons ---
    generate
        for (genvar i = 0; i < NUM_INPUTS; i++) begin : input_bind
            assign stage_decl[0].data[i] = in_data[i];
        end

        for (genvar s = 0; s < STAGES; s++) begin : tree_level
            localparam int NEXT_WIDTH = NUM_INPUTS >> (s + 1);
            
            for (genvar i = 0; i < NEXT_WIDTH; i++) begin : comp_block
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        stage_decl[s+1].data[i] <= '0;
                    end else begin
                        if (IS_SIGNED) begin
                            // Native signed comparison
                            if (stage_decl[s].data[2*i] >= stage_decl[s].data[2*i+1])
                                stage_decl[s+1].data[i] <= stage_decl[s].data[2*i];
                            else
                                stage_decl[s+1].data[i] <= stage_decl[s].data[2*i+1];
                        end else begin
                            // Cast to unsigned for comparison
                            if ($unsigned(stage_decl[s].data[2*i]) >= $unsigned(stage_decl[s].data[2*i+1]))
                                stage_decl[s+1].data[i] <= stage_decl[s].data[2*i];
                            else
                                stage_decl[s+1].data[i] <= stage_decl[s].data[2*i+1];
                        end
                    end
                end
            end
        end
    endgenerate

    // --- 3. Valid Signal Pipeline ---
    logic [STAGES:0] v_pipe; 

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_pipe <= '0;
        end else begin
            v_pipe <= {v_pipe[STAGES-1:0], valid_in};
        end
    end

    // --- 4. Final Assignments ---
    assign max_out   = stage_decl[STAGES].data[0];
    assign valid_out = v_pipe[STAGES-1];

endmodule