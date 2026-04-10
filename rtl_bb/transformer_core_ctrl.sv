`timescale 1ns/1ps

module transformer_core_ctrl import transformer_ctrl_pkg::*; #(
    parameter int MAX_LATENCY = 2 // Depth of your piped_max tree
)(
    input  logic             clk,
    input  logic             rst_n,

    // ==========================================
    // SOFTWARE / HOST INTERFACE
    // ==========================================
    input  logic             host_req_valid,
    input  opcode_e          host_opcode,
    output logic             host_req_ready,
    
    output logic             host_done_pulse, // Interrupt/Flag to Host

    // ==========================================
    // DATAPATH CONTROL BUS
    // ==========================================
    output micro_op_t        ctrl_bus,
    
    // Explicit Data Flow Controls (Not in static microcode)
    output logic             dp_in_valid,
    output logic             dp_mac_in_valid
);

    // ---------------------------------------------------------
    // 1. THE MICROCODE ROM (Combinational Lookup)
    // ---------------------------------------------------------
    micro_op_t current_uop;

    always_comb begin
        current_uop = MICRO_OP_IDLE; // Default safe state

        case (host_opcode)
            OP_LINEAR_MVM: begin
                current_uop.mac_clear_acc = 1'b1;
                current_uop.mac_op_mode   = 2'b00; // SS
            end
            
            OP_GELU_P1: begin
                current_uop.mux_tr_vec_sel = 2'b01; 
                current_uop.mux_a_sel      = 2'b01; 
                current_uop.mux_b_sel      = 3'b001; 
                current_uop.mac_elemwise   = 1'b1;
                current_uop.mac_clear_acc  = 1'b1;
                current_uop.mac_op_mode    = 2'b10; // UU
            end

            OP_GELU_P2: begin
                current_uop.mux_tr_vec_sel = 2'b10; // Feedback
                current_uop.tr_exp_sel     = 1'b1;  // Log mode
                current_uop.tr_shift_mode  = 2'b01; // Reciprocal
                current_uop.mux_a_sel      = 2'b01; 
                current_uop.mux_b_sel      = 3'b001; 
                current_uop.mac_elemwise   = 1'b1;
                current_uop.mac_clear_acc  = 1'b1;
                current_uop.mac_op_mode    = 2'b10; // UU
            end

            OP_GELU_P3: begin
                current_uop.mux_b_sel      = 3'b011; // Sigmoid Trick
                current_uop.gelu_mode      = 1'b1;  
                current_uop.mac_elemwise   = 1'b1;
                current_uop.mac_clear_acc  = 1'b1;
                current_uop.mac_op_mode    = 2'b01; // SU Mode
            end
            
            OP_SMAX_P1: begin // Summation: e^(x - max) -> out_dot
                current_uop.mux_tr_vec_sel = 2'b00; // Max-Sub Mode
                current_uop.tr_exp_sel     = 1'b0;  // Pure Exponential
                current_uop.tr_lane0_mode  = 1'b0;  // Vector Mode
                current_uop.tr_shift_mode  = 2'b00; // Bypass
                current_uop.mux_a_sel      = 2'b01; // TR Anchor
                current_uop.mux_b_sel      = 3'b001; // TR Mantissa
                current_uop.mac_elemwise   = 1'b0;  // Dot-Product / Summation Mode!
                current_uop.mac_clear_acc  = 1'b1;  // Clear sum
                current_uop.mac_op_mode    = 2'b10; // UU Mode (Exponentials are positive)
                current_uop.save_sum       = 1'b1;  // Latch the denominator sum
            end

            OP_SMAX_P2: begin // Buffer: e^(x - max) -> out_vec
                current_uop.mux_tr_vec_sel = 2'b00; // Max-Sub Mode
                current_uop.tr_exp_sel     = 1'b0;  // Log Mode
                current_uop.tr_lane0_mode  = 1'b0;  
                current_uop.tr_shift_mode  = 2'b00;
                current_uop.mux_a_sel      = 2'b01; 
                current_uop.mux_b_sel      = 3'b001; 
                current_uop.mac_elemwise   = 1'b1;  // ELEMENT-WISE (Save to out_vec)
                current_uop.mac_clear_acc  = 1'b1;
                current_uop.mac_op_mode    = 2'b10; // UU Mode
            end

            OP_SMAX_P3: begin // Broadcast Divide: out_vec * (1/out_dot)
                current_uop.tr_lane0_mode  = 1'b1;  // Scalar Mode
                current_uop.tr_shift_mode  = 2'b01; // Reciprocal (-x)
                current_uop.tr_exp_sel     = 1'b1;  // Log Mode
                current_uop.mux_a_sel      = 2'b10; // Buffered Exponentials (out_vec Pass 1)
                current_uop.mux_b_sel      = 3'b010; // Broadcast Scalar (1/out_dot)
                current_uop.mac_elemwise   = 1'b1;  // Element-wise multiply
                current_uop.mac_clear_acc  = 1'b1;
                current_uop.mac_op_mode    = 2'b10; // UU Mode (Both are positive)
            end

            OP_LN_P1: begin // Calculate Mean: Sum(X)
                current_uop.mux_a_sel      = 2'b00;  // Raw X
                current_uop.mux_b_sel      = 3'b101; // Multiply by 1.0
                current_uop.mac_elemwise   = 1'b0;   // Dot Product / Summation
                current_uop.mac_clear_acc  = 1'b1;
                current_uop.mac_op_mode    = 2'b00;  // SS Mode
                current_uop.save_mean      = 1'b1;   // Latch into saved_mean
            end

            OP_LN_P2: begin // Calculate Variance: Sum((X - mu)^2)
                current_uop.mux_sub_val_sel = 1'b1;  // Subtract Mean
                current_uop.mux_a_sel      = 2'b11;  // a_sub
                current_uop.mux_b_sel      = 3'b100; // a_sub
                current_uop.mac_elemwise   = 1'b0;   // Dot Product / Summation
                current_uop.mac_clear_acc  = 1'b1;
                current_uop.mac_op_mode    = 2'b00;  // SS Mode
                current_uop.save_sum       = 1'b1;   // Latch into saved_sum (for TR Array)
            end

            OP_LN_P3: begin // Final Normalization: (X - mu) * ISD
                current_uop.mux_sub_val_sel = 1'b1; // Keep Mean subtraction active
                current_uop.tr_lane0_mode  = 1'b1;   
                current_uop.tr_shift_mode  = 2'b10;  // ISD Mode (Triggers the Var/N shift!)
                current_uop.tr_exp_sel     = 1'b1;   

                // Read a_sub directly! No buffering required.
                current_uop.mux_a_sel      = 2'b11;  
                current_uop.mux_b_sel      = 3'b010; // scalar_recip_8b (Holds ISD)
                current_uop.mac_elemwise   = 1'b1;   
                current_uop.mac_clear_acc  = 1'b1;
                current_uop.mac_op_mode    = 2'b01;  // SU Mode
            end
        endcase
    end

    // ---------------------------------------------------------
    // 2. THE SEQUENCER STATE MACHINE
    // ---------------------------------------------------------
    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LOAD,
        ST_WAIT_PIPE,
        ST_PULSE_MAC,
        ST_WAIT_MAC
    } state_e;

    state_e state_q, state_d;
    logic [3:0] wait_cnt_q, wait_cnt_d;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q    <= ST_IDLE;
            wait_cnt_q <= '0;
        end else begin
            state_q    <= state_d;
            wait_cnt_q <= wait_cnt_d;
        end
    end

    always_comb begin
        // Default assignments
        state_d         = state_q;
        wait_cnt_d      = wait_cnt_q;
        host_req_ready  = 1'b0;
        host_done_pulse = 1'b0;
        
        dp_in_valid     = 1'b0;
        dp_mac_in_valid = 1'b0;
        ctrl_bus        = MICRO_OP_IDLE;

        case (state_q)
            ST_IDLE: begin
                host_req_ready = 1'b1;
                if (host_req_valid) begin
                    state_d = ST_LOAD;
                end
            end

            ST_LOAD: begin
                ctrl_bus    = current_uop; 
                dp_in_valid = 1'b1; // Push data into pipeline
                wait_cnt_d  = MAX_LATENCY;
                state_d     = ST_WAIT_PIPE;
            end

            ST_WAIT_PIPE: begin
                ctrl_bus = current_uop; // Hold control signals stable
                dp_in_valid = 1'b1;

                if (wait_cnt_q == 0) begin
                    state_d = ST_PULSE_MAC;
                end else begin
                    wait_cnt_d = wait_cnt_q - 1;
                end
            end

            ST_PULSE_MAC: begin
                ctrl_bus        = current_uop;
                dp_mac_in_valid = 1'b1; // Trigger the multiplier
                wait_cnt_d      = 4'd3; // Wait for MAC to settle + feedback routing
                state_d         = ST_WAIT_MAC;
            end

            ST_WAIT_MAC: begin
                ctrl_bus = current_uop; 
                if (wait_cnt_q == 0) begin
                    host_done_pulse = 1'b1; // Tell software we finished this pass!
                    state_d         = ST_IDLE;
                end else begin
                    wait_cnt_d = wait_cnt_q - 1;
                end
            end
        endcase
    end

endmodule