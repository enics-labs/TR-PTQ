`timescale 1ns / 1ps
`define POST_USE_TR_LN
// `define POST_USE_DIV

`ifdef POST_USE_TR_LN
`ifdef POST_USE_DIV
    initial begin
        $error("Define only one of POST_USE_TR_LN or POST_USE_DIV in soft_dsp_eng_post_wrapper.sv");
    end
`endif
`endif

module soft_dsp_eng_post_wrapper #(
    parameter int N          = 16,
    parameter int W          = 8,
    parameter int ACC_W      = 32,
    parameter int FRAC       = 4,
    parameter int ITER       = 2,
    parameter int POST_IN_W  = 17,
    parameter int LN_BITS    = 4,
    parameter int RECIP_BITS = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // Input stream to the chunk-level engine
    input  logic                    in_valid,
    output logic                    in_ready,
    input  logic [1:0]              op_mode,
    input  logic                    mode_elemwise,
    input  logic                    dsp_mode,
    input  logic signed [W-1:0]     a [N],
    input  logic signed [W-1:0]     b [N],
    input  logic                    clear_acc,
    input  logic                    vector_last_in,

    // Raw engine outputs
    // TODO: Add busy signal (or take care of in_ready or both)
    output logic                    out_valid,
    input  logic                    out_ready,
    output logic [N-1:0]            out_valid_mask,
    output logic signed [ACC_W-1:0] out_vec [N],
    output logic signed [ACC_W-1:0] out_dot,

    // Post-processing result
    output logic                    post_valid,
    output logic                    post_busy,
    output logic signed [ACC_W-1:0] post_out
);

    // ========================================================================
    // STAGE 1: Pipelined Max Tree & Delay Lines (SoftMax Path)
    // ========================================================================
    localparam int MAX_LATENCY = $clog2(N) - 1;
    
    logic signed [W-1:0] x_max;
    logic                max_valid;

    piped_max #(
        .NUM_INPUTS(N),
        .DATA_WIDTH(W),
        .IS_SIGNED(1)
    ) u_piped_max (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (in_valid),
        .in_data   (a),
        .max_out   (x_max),
        .valid_out (max_valid)
    );

    // We must delay 'a' and all control signals to match the piped_max latency
    logic signed [W-1:0] a_d                [MAX_LATENCY+1][N];
    logic                clear_acc_d        [MAX_LATENCY+1];
    logic                vector_last_in_d   [MAX_LATENCY+1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s <= MAX_LATENCY; s++) begin
                clear_acc_d[s]      <= 1'b0;
                vector_last_in_d[s] <= 1'b0;
                for (int i = 0; i < N; i++) a_d[s][i] <= '0;
            end
        end else begin
            clear_acc_d[0]      <= clear_acc;
            vector_last_in_d[0] <= vector_last_in;
            for (int i = 0; i < N; i++) a_d[0][i] <= a[i];
            
            for (int s = 1; s <= MAX_LATENCY; s++) begin
                clear_acc_d[s]      <= clear_acc_d[s-1];
                vector_last_in_d[s] <= vector_last_in_d[s-1];
                for (int i = 0; i < N; i++) a_d[s][i] <= a_d[s-1][i];
            end
        end
    end

    // ========================================================================
    // STAGE 2: Max Subtraction & TR-Decomposition
    // ========================================================================
    logic signed [W-1:0] a_sub [N];

    max_sub #(
        .NUM_INPUTS(N),
        .DATA_WIDTH(W),
        .CLAMP_MIN(-(1 << (W-1)))
    ) u_max_sub (
        .in_data  (a_d[MAX_LATENCY]),
        .x_max    (x_max),
        .out_data (a_sub)
    );

    logic [W-1:0] exp_a_mux [N];
    logic [W-1:0] exp_b_mux [N];

    tr_exp_wrapper #(
        .N(N),
        .FRAC(FRAC),
        .ITER(ITER)
    ) u_tr_exp_wrapper (
        .dsp_mode (1'b1),     // Hardwired to 1 (we do the muxing at the top level)
        .a        (a_sub),    // Feed the shifted values
        .b        (a_sub),    // Unused by TR-exp
        .a_mux    (exp_a_mux),// Outputs e_a
        .b_mux    (exp_b_mux) // Outputs e_frac
    );

    // ========================================================================
    // STAGE 3: Master Datapath Multiplexer (Memory Bypass vs SoftMax)
    // ========================================================================
    logic [W-1:0] mac_a [N];
    logic [W-1:0] mac_b [N];
    logic                mac_valid;
    logic                mac_clear_acc;
    logic                mac_last_in;

    always_comb begin
        int i;
        if (dsp_mode) begin
            // SoftMax Path: Use the decomposed exponentials and the delayed control signals
            for (i = 0; i < N; i++) begin
                mac_a[i] = exp_a_mux[i];
                mac_b[i] = exp_b_mux[i];
            end
            mac_valid     = max_valid;
            mac_clear_acc = clear_acc_d[MAX_LATENCY];
            mac_last_in   = vector_last_in_d[MAX_LATENCY];
        end else begin
            // Direct Memory Path: 0 latency bypass
            for (i = 0; i < N; i++) begin
                mac_a[i] = a[i];
                mac_b[i] = b[i];
            end
            mac_valid     = in_valid;
            mac_clear_acc = clear_acc;
            mac_last_in   = vector_last_in;
        end
    end

    // ========================================================================
    // STAGE 4: Vector Multiply-Accumulate (MAC) Engine
    // ========================================================================
    vec_mul #(
        .N(N),
        .W(W),
        .ACC_W(ACC_W)
    ) dsp_mul (
        .clk           (clk),
        .rst_n         (rst_n),
        .in_valid      (mac_valid),
        .in_ready      (in_ready),
        .op_mode       (op_mode),
`ifndef DOT_PROD_ONLY
        .mode_elemwise (mode_elemwise),
`else
        .mode_elemwise (1'b0),
`endif
        .a             (mac_a),
        .b             (mac_b),
        .clear_acc     (mac_clear_acc),
        .out_valid     (out_valid),
        .out_ready     (out_ready),
        .out_valid_mask(out_valid_mask),
        .out_vec       (out_vec)
    );

    assign out_dot = out_vec[0];

    // ========================================================================
    // STAGE 5: Chunk-Level Synchronization Pipeline
    // ========================================================================
    localparam int MAC_PIPELINE_DEPTH = $clog2(N) + 1;
    logic [MAC_PIPELINE_DEPTH-1:0] last_chunk_sr;
    logic advance;

    assign advance = (~out_valid) || (out_valid && out_ready);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            last_chunk_sr <= '0;
        end else if (advance) begin
            // Shift the tracking bit through the parameterized register
            last_chunk_sr <= {last_chunk_sr[MAC_PIPELINE_DEPTH-2:0], 
                              (mac_valid && in_ready && mac_last_in)};
        end
    end

    logic final_sum_fire;
    assign final_sum_fire = out_valid && out_ready && last_chunk_sr[MAC_PIPELINE_DEPTH-1];

    logic [POST_IN_W-1:0] final_sum_reg;

    // ========================================================================
    // STAGE 6: Post-Processing (Logarithm or Division)
    // ========================================================================
`ifdef POST_USE_TR_LN
    logic                 ln_pending;
    wire signed [POST_IN_W/2-1:0] ln_yq;

    tr_ln #(
        .WIDTH(POST_IN_W),
        .BITS(LN_BITS)
    ) u_tr_ln (
        .xq(final_sum_reg),
        .yq(ln_yq)
    );

`elsif POST_USE_DIV
    logic                 div_start;
    logic                 div_busy;
    logic                 div_done;
    logic                 div_valid;
    logic                 div_dbz;
    logic [POST_IN_W-1:0] div_val;
    logic [POST_IN_W-1:0] div_rem;

    divu_int #(
        .WIDTH(POST_IN_W)
    ) u_divu_int (
        .clk  (clk),
        .rst_n(rst_n),
        .start(div_start),
        .busy (div_busy),
        .done (div_done),
        .valid(div_valid),
        .dbz  (div_dbz),
        .a    ({{(POST_IN_W-(RECIP_BITS+1)){1'b0}}, 1'b1, {RECIP_BITS{1'b0}}}),
        .b    (final_sum_reg),
        .val  (div_val),
        .rem  (div_rem)
    );
`else
    initial begin
        $error("Define exactly one of POST_USE_TR_LN or POST_USE_DIV in soft_dsp_eng_post_wrapper.sv");
    end
`endif

    // ========================================================================
    // STAGE 7: Post-Processing Controller FSM
    // ========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            final_sum_reg <= '0;
            post_valid    <= 1'b0;
            post_out      <= '0;
`ifdef POST_USE_TR_LN
            ln_pending    <= 1'b0;
`elsif POST_USE_DIV
            div_start     <= 1'b0;
`endif
        end else begin
            post_valid <= 1'b0;
`ifdef POST_USE_DIV
            div_start  <= 1'b0;
`endif

            if (final_sum_fire) begin
                final_sum_reg <= out_dot[POST_IN_W-1:0];
`ifdef POST_USE_TR_LN
                ln_pending <= 1'b1;
`elsif POST_USE_DIV
                div_start  <= 1'b1;
`endif
            end

`ifdef POST_USE_TR_LN
            else if (ln_pending) begin
                ln_pending <= 1'b0;
                post_valid <= 1'b1;
                post_out   <= {{(ACC_W-(POST_IN_W/2)){ln_yq[POST_IN_W/2-1]}}, ln_yq};
            end
`endif

`ifdef POST_USE_DIV
            if (div_done) begin
                post_valid <= 1'b1;
                post_out   <= {{(ACC_W-POST_IN_W){1'b0}}, div_val};
            end
`endif
        end
    end

`ifdef POST_USE_TR_LN
    assign post_busy = ln_pending;
`elsif POST_USE_DIV
    assign post_busy = div_busy;
`else
    assign post_busy = 1'b0;
`endif

endmodule