/*
 * @module   tr_backbone_wrapper
 * @brief    Encapsulated Log-Domain Math Pipeline (Vectorized).
 * @details  Chains pre-LN modification, TR-ln, post-LN modification, and TR-exp.
 *           Purely combinational. Expects wide inputs and natively steps down 
 *           precision through the tr_ln_alu block.
 *
 * @param    N           Vector dimension.
 * @param    WIDTH_IN    Incoming datapath width (e.g., 16-bit).
 * @param    WIDTH_OUT   Target operating width for EXP/Mantissa (e.g., 8-bit).
 * @param    FRAC_W      Fractional bit width.
 * @param    LUT_IDX_W   Width of the ROM index for the Taylor anchor.
 */
module tr_backbone_wrapper #(
    parameter int N         = 8,
    parameter int WIDTH_IN  = 16,
    parameter int WIDTH_OUT = 8,
    parameter int FRAC_W    = 4,
    parameter int LUT_IDX_W = 3
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    in_valid,
    output logic                    out_valid,

    // Control Flags
    input  logic                    mode_pre_ln,  // 0: Bypass, 1: Add +1.0
    input  logic [1:0]              mode_post_ln, // 00: By, 01: -1.0x, 10: -0.5x
    
    // Datapath Input
    input  logic signed [WIDTH_IN-1:0]  vec_in [N],
    
    // Datapath Outputs
    output logic      [LUT_IDX_W-1:0]   a_idx_out [N],
    output logic      [WIDTH_OUT-1:0]   mantisa_out [N],
    output logic                        is_zero_out [N]
);

    // =========================================================
    // STAGE 1: PRE-LN (Combinational)
    // =========================================================
    logic signed [WIDTH_IN-1:0]  pre_ln_out  [N];

    pre_ln_modifier #(
        .N(N), 
        .WIDTH_IN(WIDTH_IN), 
        .FRAC_W(FRAC_W)
    ) u_pre_ln (
        .x_in         (vec_in),
        .mode_add_one (mode_pre_ln),
        .y_out        (pre_ln_out)
    );

    // --- PIPELINE REGISTER 1 ---
    logic signed [WIDTH_IN-1:0] s1_pre_ln_reg [N];
    logic [1:0]                 s1_mode_post_ln;
    logic                       s1_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
            s1_mode_post_ln <= '0;
            for(int i=0; i<N; i++) s1_pre_ln_reg[i] <= '0;
        end else begin
            s1_valid <= in_valid;
            s1_mode_post_ln <= mode_post_ln; // Carry the control flag forward
            s1_pre_ln_reg <= pre_ln_out;
        end
    end

    // =========================================================
    // STAGE 2: TR-LN & POST-LN (Combinational)
    // =========================================================
    logic signed [WIDTH_OUT-1:0] tr_ln_out   [N];
    logic signed [WIDTH_OUT-1:0] post_ln_out [N];

    generate
        for (genvar i = 0; i < N; i++) begin : gen_tr_lanes
            
            // TR-LN ALU (Steps down WIDTH_IN -> WIDTH_OUT)
            tr_ln_alu #(
                .WIDTH(WIDTH_IN), 
                .BITS(FRAC_W),          // Note: tr_ln_alu uses "BITS" for fractional width
                .OUT_WIDTH(WIDTH_OUT)
            ) u_tr_ln (
                .xq (s1_pre_ln_reg[i]),    // Note: tr_ln_alu expects unsigned/positive input 'xq'
                .yq (tr_ln_out[i])
            );

        end
    endgenerate

    post_ln_modifier #(
        .N(N), 
        .W(WIDTH_OUT)
    ) u_post_ln (
        .x_in     (tr_ln_out),
        .mode_sel (s1_mode_post_ln),
        .y_out    (post_ln_out)
    );

    // --- PIPELINE REGISTER 2 ---
    logic signed [WIDTH_OUT-1:0] s2_post_ln_reg [N];
    logic                        s2_valid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
            for(int j=0; j<N; j++) s2_post_ln_reg[j] <= '0;
        end else begin
            s2_valid <= s1_valid;
            s2_post_ln_reg <= post_ln_out;
        end
    end

    // =========================================================
    // STAGE 3: TR-EXP (Combinational to Output)
    // =========================================================
    assign out_valid = s2_valid;

    generate
        for (genvar i = 0; i < N; i++) begin : gen_tr_exp_lanes
            
            // 4. TR-EXP ALU
            tr_exp_alu #(
                .WIDTH(WIDTH_OUT),
                .FRAC_W(FRAC_W),
                .LUT_IDX_W(LUT_IDX_W),
                .ITER(2) // Quadratic
            ) u_tr_exp (
                .x       (s2_post_ln_reg[i]),
                .a_idx   (a_idx_out[i]),
                .mantisa (mantisa_out[i]),
                .is_zero (is_zero_out[i])
            );
        end
    endgenerate

endmodule