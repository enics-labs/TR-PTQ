// assuming power of two vector size

module online_sum #(
    parameter int NUM_INPUTS    = 8,
    parameter int DATA_WIDTH    = 8,
    parameter int FRAC_W        = 4,
    parameter int LUT_IDX_W     = 3,
    parameter int ITER          = 1,
    parameter int MODE          = 0 // 0: Int Max+Sub | 1: Ext Max, Int Sub | 2: Ext Max+Sub (Direct Exp)
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          valid_in,
    input  logic signed [DATA_WIDTH-1:0]  in_data [NUM_INPUTS],
    input  logic signed [DATA_WIDTH-1:0]  ext_x_max, // Used if MODE == 1

    output logic                          valid_out,
    output logic signed [DATA_WIDTH-1:0]  e_a [NUM_INPUTS],
    output logic signed [DATA_WIDTH-1:0]  e_frac [NUM_INPUTS]
);

    // ========================================================================
    // STAGE 1, 2, 3: Pre-processing (Max Tree & Subtraction)
    // ========================================================================
    logic signed [DATA_WIDTH-1:0] x_shifted_clamped [NUM_INPUTS];
    logic                         active_valid;

    generate
        if (MODE == 0) begin : gen_mode_0_int_max_int_sub
            localparam int MAX_LATENCY = $clog2(NUM_INPUTS) - 1;

            logic signed [DATA_WIDTH-1:0] x_max;
            logic                         max_valid;

            piped_max #(
                .NUM_INPUTS (NUM_INPUTS),
                .DATA_WIDTH (DATA_WIDTH)
            ) u_max_tree (
                .clk      (clk),
                .rst_n    (rst_n),
                .valid_in (valid_in),
                .in_data  (in_data),
                .max_out  (x_max),
                .valid_out(max_valid)
            );

            logic signed [DATA_WIDTH-1:0] in_data_d [MAX_LATENCY+1][NUM_INPUTS];
            logic                         valid_d   [MAX_LATENCY+1];

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int s = 0; s < MAX_LATENCY; s++)
                        valid_d[s] <= 1'b0;
                    
                    for (int i3 = 0; i3 < MAX_LATENCY+1; i3++)
                        for (int i2 = 0; i2 < NUM_INPUTS; i2++)
                            in_data_d[i3][i2] <= 8'd0;
                end else begin
                    valid_d[0] <= valid_in;
                    for (int i = 0; i < NUM_INPUTS; i++)
                        in_data_d[0][i] <= in_data[i];

                    for (int s = 1; s < MAX_LATENCY+1; s++) begin
                        valid_d[s] <= valid_d[s-1];
                        for (int i = 0; i < NUM_INPUTS; i++)
                            in_data_d[s][i] <= in_data_d[s-1][i];
                    end
                end
            end

            max_sub #(
                .NUM_INPUTS(NUM_INPUTS),
                .DATA_WIDTH(DATA_WIDTH)
            ) u_max_sub (
                .in_data  (in_data_d[MAX_LATENCY]),
                .x_max    (x_max),
                .out_data (x_shifted_clamped)
            );
            
            assign active_valid = valid_d[MAX_LATENCY-1];
        
        end else if (MODE == 1) begin : gen_mode_1_ext_max_int_sub
            logic signed [DATA_WIDTH-1:0] in_data_d [NUM_INPUTS];
            logic signed [DATA_WIDTH-1:0] ext_x_max_d;
            
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < NUM_INPUTS; i++) in_data_d[i] <= '0;
                    ext_x_max_d <= '0;
                end else begin
                    for (int i = 0; i < NUM_INPUTS; i++) in_data_d[i] <= in_data[i];
                    ext_x_max_d <= ext_x_max;
                end
            end

            max_sub #(
                .NUM_INPUTS(NUM_INPUTS),
                .DATA_WIDTH(DATA_WIDTH)
            ) u_max_sub (
                .in_data  (in_data_d),
                .x_max    (ext_x_max_d),
                .out_data (x_shifted_clamped)
            );
            
            assign active_valid = valid_in;

        end else begin : gen_mode_2_ext_max_ext_sub
            logic signed [DATA_WIDTH-1:0] in_data_d [NUM_INPUTS];
            
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int i = 0; i < NUM_INPUTS; i++) in_data_d[i] <= '0;
                end else begin
                    for (int i = 0; i < NUM_INPUTS; i++) in_data_d[i] <= in_data[i];
                end
            end
            
            assign x_shifted_clamped = in_data_d;
            assign active_valid = valid_in;
        end
    endgenerate

    // Output Valid Synchronization
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_out <= 1'b0;
        else        valid_out <= active_valid;
    end

    // ========================================================================
    // STAGE 4: TR-EXP Decomposition per element (fully parallel)
    // ========================================================================
    generate
        for (genvar g = 0; g < NUM_INPUTS; g++) begin : GEN_EXP
            tr_exp #(
                .WIDTH     (DATA_WIDTH),
                .FRAC_W    (FRAC_W),
                .LUT_IDX_W (LUT_IDX_W),
                .ITER      (ITER)
            ) u_tr_exp (
                .x       (x_shifted_clamped[g]), // signed Q4, ≤ 0
                .e_a     (e_a[g]),               // Q1.7 or Q8
                .mantisa (e_frac[g]),            // Q1.7 or Q8
                .is_zero ()
            );
        end
    endgenerate

endmodule
