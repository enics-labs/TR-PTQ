`timescale 1ns/1ps

module transformer_core_datapath #(
    parameter int N          = 8,
    parameter int W          = 8,
    parameter int ACC_W      = 32,
    parameter int FRAC       = 4
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // ==========================================
    // EXTERNAL DATA INTERFACE
    // ==========================================
    input  logic                    in_valid,
    input  logic signed [W-1:0]     a [N],
    input  logic signed [W-1:0]     b [N],
    
    output logic                    out_valid,
    output logic signed [ACC_W-1:0] out_vec [N],
    output logic signed [ACC_W-1:0] out_dot,

    // ==========================================
    // THE CONTROL BUS (Driven by Future FSM)
    // ==========================================
    // TR Array Controls
    input  logic                    ctrl_tr_lane0_mode,
    input  logic [1:0]              ctrl_tr_shift_mode,
    input  logic                    ctrl_tr_exp_sel,
    // Routing MUX Controls
    input  logic                    ctrl_mux_tr_vec_sel, // 0: Max_Sub, 1: Delayed A
    input  logic [1:0]              ctrl_mux_a_sel,      // 00: Mem A, 01: TR_EA, 10: Buffered Out
    input  logic [1:0]              ctrl_mux_b_sel,      // 00: Mem B, 01: TR_MAN, 10: Broadcast Scalar
    // MAC Engine Controls
    input  logic [1:0]              ctrl_mac_op_mode,
    input  logic                    ctrl_mac_elemwise,
    input  logic                    ctrl_mac_clear_acc,
    input  logic                    ctrl_mac_in_valid,
    input  logic                    ctrl_save_sum
);

    // ========================================================================
    // STAGE 1: Pipelined Max Tree & Data Delay Lines
    // ========================================================================
    localparam int MAX_LATENCY = $clog2(N) - 1;
    
    logic signed [W-1:0] x_max;
    logic                max_valid;

    piped_max #(
        .NUM_INPUTS(N), 
        .DATA_WIDTH(W)
    ) u_piped_max (
        .clk        (clk), 
        .rst_n      (rst_n),
        .valid_in   (in_valid), 
        .in_data    (a),
        .max_out    (x_max), 
        .valid_out  (max_valid)
    );

    // Delay incoming memory vectors to match the Max Tree latency
    logic signed [W-1:0] a_d [MAX_LATENCY+1][N];
    logic signed [W-1:0] b_d [MAX_LATENCY+1][N];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int s = 0; s <= MAX_LATENCY; s++) begin
                for (int i = 0; i < N; i++) begin
                    a_d[s][i] <= '0;
                    b_d[s][i] <= '0;
                end
            end
        end else begin
            for (int i = 0; i < N; i++) begin
                a_d[0][i] <= a[i];
                b_d[0][i] <= b[i];
            end
            for (int s = 1; s <= MAX_LATENCY; s++) begin
                for (int i = 0; i < N; i++) begin
                    a_d[s][i] <= a_d[s-1][i];
                    b_d[s][i] <= b_d[s-1][i];
                end
            end
        end
    end

    wire signed [W-1:0] a_delayed [N];
    wire signed [W-1:0] b_delayed [N];
    assign a_delayed = a_d[MAX_LATENCY];
    assign b_delayed = b_d[MAX_LATENCY];

    // ========================================================================
    // STAGE 2: Max Subtraction & TR Input Routing
    // ========================================================================
    logic signed [W-1:0] a_sub [N];
    logic signed [W-1:0] tr_vec_in [N];

    max_sub #(
        .NUM_INPUTS(N), 
        .DATA_WIDTH(W)
    ) u_max_sub (
        .in_data(a_delayed), 
        .x_max(x_max), 
        .out_data(a_sub)
    );

    // MUX: Feed TR Array with either Max_Sub (SoftMax) or Raw Data (GELU)
    always_comb begin
        for (int i = 0; i < N; i++) begin
            tr_vec_in[i] = ctrl_mux_tr_vec_sel ? a_delayed[i] : a_sub[i];
        end
    end

    // ========================================================================
    // STAGE 3: The Non-Linear Engine (TR Heterogeneous Array)
    // ========================================================================
    // Scalar capture register
    logic [19:0] saved_sum;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) saved_sum <= '0;
        else if (ctrl_save_sum) saved_sum <= out_dot[19:0];
    end

    wire [19:0] aligned_sum = saved_sum >> 4;

    logic [7:0]  y_ea_vec  [N];
    logic [7:0]  y_man_vec [N];
    logic [19:0] y_ea_scalar;
    logic [19:0] y_man_scalar;

    tr_hetero_array #(.N(N)) u_tr_array (
        .x_vec_in         (tr_vec_in),
        .x_scalar_in      (aligned_sum), // Fed back from the MAC accumulator (ALIGNMENT FIX: Q4.12 -> Qx.8)
        .lane_0_mode      (ctrl_tr_lane0_mode),
        .shift_mode       (ctrl_tr_shift_mode),
        .exp_in_sel       (ctrl_tr_exp_sel),
        
        .y_ea_vec_out     (y_ea_vec),
        .y_man_vec_out    (y_man_vec),
        .y_ea_scalar_out  (y_ea_scalar),
        .y_man_scalar_out (y_man_scalar)
    );

    // ========================================================================
    // STAGE 4: Scalar Math Reflector (Broadcast Generator)
    // ========================================================================
    // Since Option A uses a split bus, we manually calculate the single 
    // scalar reciprocal here and cast it to Q4.4 to broadcast to the MAC Engine.
    logic [39:0] scalar_mult_raw;
    logic [19:0] scalar_recip_20b;
    logic [7:0]  scalar_recip_8b;

    assign scalar_mult_raw  = y_ea_scalar * y_man_scalar;
    assign scalar_recip_20b = scalar_mult_raw[27:8]; // Restore Q12.8 after mult
    assign scalar_recip_8b  = (scalar_mult_raw + 40'd2048) >> 12; // Cast Q4.12 -> Q4.4

    // ========================================================================
    // STAGE 5: MAC Engine Datapath Multiplexers
    // ========================================================================
    logic [W-1:0] mac_a_in [N];
    logic [W-1:0] mac_b_in [N];

    always_comb begin
        for (int i = 0; i < N; i++) begin
            // MUX A
            case (ctrl_mux_a_sel)
                2'b00: mac_a_in[i] = a_delayed[i];                // Raw Memory X
                2'b01: mac_a_in[i] = y_ea_vec[i];                 // TR Array Anchor
                2'b10: mac_a_in[i] = (out_vec[i] + 32'd128) >> 8; // Feedback Buffered MAC Output
                default: mac_a_in[i] = a_delayed[i];
            endcase

            // MUX B
            case (ctrl_mux_b_sel)
                2'b00: mac_b_in[i] = b_delayed[i];                // Raw Memory W
                2'b01: mac_b_in[i] = y_man_vec[i];                // TR Array Mantissa
                2'b10: mac_b_in[i] = scalar_recip_8b;             // Broadcast Scalar (e.g. 1/Sum)
                2'b11: mac_b_in[i] = b_delayed[i];                // 1.0 Constant in Q4.4
            endcase
        end
    end

    // ========================================================================
    // STAGE 6: Vector Multiply-Accumulate (MAC) Engine
    // ========================================================================
    // Using your existing vec_mul module
    vec_mul #(
        .N(N), .W(W), .ACC_W(ACC_W)
    ) u_mac_engine (
        .clk            (clk),
        .rst_n          (rst_n),
        .in_valid       (ctrl_mac_in_valid), // Gated by TB/FSM
        .in_ready       (),     
        .op_mode        (ctrl_mac_op_mode),
        .mode_elemwise  (ctrl_mac_elemwise),
        .a              (mac_a_in),
        .b              (mac_b_in),
        .clear_acc      (ctrl_mac_clear_acc),
        
        .out_valid      (out_valid),
        .out_ready      (1'b1), 
        .out_valid_mask (),
        .out_vec        (out_vec)
    );

    assign out_dot = out_vec[0];

endmodule