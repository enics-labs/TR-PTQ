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
    input  logic                    ctrl_mux_sub_val_sel,// 0: Max, 1: Mean
    input  logic [1:0]              ctrl_mux_tr_vec_sel, // 0: Max_Sub, 1: Delayed A
    input  logic [1:0]              ctrl_mux_a_sel,      // 00: Mem A, 01: TR_EA, 10: Buffered Out
    input  logic [2:0]              ctrl_mux_b_sel,      // 00: Mem B, 01: TR_MAN, 10: Broadcast Scalar
    // MAC Engine Controls
    input  logic [1:0]              ctrl_mac_op_mode,
    input  logic                    ctrl_mac_elemwise,
    input  logic                    ctrl_mac_clear_acc,
    input  logic                    ctrl_mac_in_valid,
    input  logic                    ctrl_save_sum,
    input  logic                    ctrl_save_mean,
    input  logic                    ctrl_gelu_mode
);

    // ========================================================================
    // DERIVED MATH CONSTANTS (Fully scalable based on W and FRAC)
    // ========================================================================
    localparam int ONE_Q_FRAC = 1 << FRAC;       // 1.0 in Q(W.FRAC)
    localparam int ONE_Q_MAX  = 1 << W;          // 1.0 in Q(0.W)
    localparam int HALF_FRAC  = 1 << (FRAC - 1); // 0.5 in Q(W.FRAC)
    localparam int ROUND_W    = 1 << (W - 1);    // 0.5 in Q(W.W)
    
    // Saturation limits
    localparam logic [W-1:0] MAX_POS = {1'b0, {(W-1){1'b1}}}; // 8'h7F (127)
    localparam logic [W-1:0] MAX_UNS = {W{1'b1}};             // 8'hFF (255)

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
    logic signed [W-1:0] a_alpha [N];
    logic signed [W-1:0] a_feedback [N]; // Feed from MAC feedback
    logic signed [W-1:0] tr_vec_in [N];

    logic signed [W-1:0] saved_mean; 
    logic signed [W-1:0] sub_val;

    // The Subtractor MUX
    assign sub_val = ctrl_mux_sub_val_sel ? saved_mean : x_max;

    // SoftMax path
    scalar_sub #(
        .NUM_INPUTS(N), 
        .DATA_WIDTH(W)
    ) u_scalar_sub (
        .in_data(a_delayed), 
        .sub_val(sub_val), 
        .out_data(a_sub)
    );

    // GELU Scaling path
    alpha_stabilizer #(
        .N(N), 
        .W(W)
    ) u_alpha_stab (
        .in_vec(a_delayed), 
        .out_vec(a_alpha)
    );

    // Feedback Conversion: Convert 32-bit MAC output to 8-bit Q4.4 for TR Array
    always_comb begin
        for (int i = 0; i < N; i++) begin
            // Shift Q4.12 back to Q4.4 with floor truncation
            // Add 1.0 (ONE_Q_FRAC) and scale by 2 (shift left 1)
            // We use (>> W) to safely extract the upper Q4.4 bits from the Q4.12 product
            a_feedback[i] = ((out_vec[i] >> W) + ONE_Q_FRAC) << 1;
        end
    end

    // MUX: Feed TR Array with either Max_Sub (SoftMax) or Raw Data (GELU)
    always_comb begin
        for (int i = 0; i < N; i++) begin
            case (ctrl_mux_tr_vec_sel) // Now a 2-bit control signal
                2'b00: tr_vec_in[i] = a_sub[i];      // SoftMax / LN Max-Sub
                2'b01: tr_vec_in[i] = a_alpha[i];    // GELU Pass 1 (EXP)
                2'b10: tr_vec_in[i] = a_feedback[i]; // GELU Pass 2 (Reciprocal)
                default: tr_vec_in[i] = a_alpha[i];
            endcase
        end
    end

    // ========================================================================
    // STAGE 3: The Non-Linear Engine (TR Heterogeneous Array)
    // ========================================================================
    // Scalar capture registers
    logic [19:0] saved_sum;

    // The total right-shift to convert a Q8.8 sum back to a Q4.4 mean
    localparam int MEAN_SHIFT = $clog2(N) + FRAC;
    localparam int MEAN_ROUND = 1 << (MEAN_SHIFT - 1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            saved_sum  <= '0;
            saved_mean <= '0;
        end else begin
            if (ctrl_save_sum)  saved_sum  <= out_dot[19:0];
            if (ctrl_save_mean) saved_mean <= $signed(out_dot + MEAN_ROUND) >>> MEAN_SHIFT;
        end
    end

    // If the FSM is calculating ISD (LayerNorm), shift by an extra log2(N) to divide by N!
    wire is_isd_mode = (ctrl_tr_shift_mode == 2'b10);
    wire [19:0] aligned_sum = is_isd_mode ? (saved_sum >> $clog2(N)) : (saved_sum >> FRAC);

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

    localparam logic [39:0] SCALAR_ROUND = 40'd1 << 11;

    assign scalar_mult_raw  = y_ea_scalar * y_man_scalar;
    assign scalar_recip_20b = scalar_mult_raw[27:8]; // Restore Q12.8 after mult
    assign scalar_recip_8b  = (scalar_mult_raw + SCALAR_ROUND) >> 12; // Cast Q4.12 -> Q4.4

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
                2'b10: mac_a_in[i] = (out_vec[i] + ROUND_W) >> W; // Feedback Buffered MAC Output
                2'b11: mac_a_in[i] = a_sub[i];                    // LN Direct Subtractor Routing
            endcase

            // MUX B
            case (ctrl_mux_b_sel)
                3'b000: mac_b_in[i] = b_delayed[i];                // Raw Memory W
                3'b001: mac_b_in[i] = y_man_vec[i];                // TR Array Mantissa
                3'b010: mac_b_in[i] = scalar_recip_8b;             // Broadcast Scalar (e.g. 1/Sum)
                3'b011: begin
                    // Shift out the fractional bits to get Q0.W precision, adding half-bit for rounding
                    logic [W+1:0] inv_S;
                    inv_S = ((out_vec[i] << 1) + HALF_FRAC) >> FRAC;
                    
                    if (ctrl_gelu_mode) begin
                        logic [W+1:0] sigmoid;

                        // Symmetry Trick: 1.0 - sigma(-x)
                        sigmoid = (a_delayed[i][W-1]) ? (ONE_Q_MAX - inv_S) : inv_S;
                        
                        // Saturate and pass full unsigned precision to the SU MAC
                        mac_b_in[i] = (sigmoid >= MAX_UNS) ? MAX_UNS : sigmoid[W-1:0];
                    end else begin
                        mac_b_in[i] = (inv_S >= MAX_UNS) ? MAX_UNS : inv_S[W-1:0];
                    end
                end
                3'b100: mac_b_in[i] = a_sub[i];    // Direct Subtractor Routing
                3'b101: mac_b_in[i] = ONE_Q_FRAC;  // Hardware 1.0 Constant
                default: mac_b_in[i] = b_delayed[i];
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