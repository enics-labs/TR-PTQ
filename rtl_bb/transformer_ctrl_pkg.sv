`timescale 1ns/1ps

package transformer_ctrl_pkg;

    // ---------------------------------------------------------
    // 1. OPERATION OPCODES
    // ---------------------------------------------------------
    typedef enum logic [3:0] {
        OP_IDLE       = 4'd0,
        
        // Linear / Basic Math
        OP_LINEAR_MVM = 4'd1,
        
        // SoftMax Sequence
        OP_SMAX_P1    = 4'd2, // Summation
        OP_SMAX_P2    = 4'd3, // Buffered Exponentials
        OP_SMAX_P3    = 4'd4, // Broadcast Divide
        
        // GELU Sequence
        OP_GELU_P1    = 4'd5, // Exponential (E)
        OP_GELU_P2    = 4'd6, // Reciprocal (1/S)
        OP_GELU_P3    = 4'd7, // Final Sigmoid Mult

        // LayerNorm Sequence
        OP_LN_P1      = 4'd8, // Sum for Mean
        OP_LN_P2      = 4'd9, // Variance Sum
        OP_LN_P3      = 4'd10 // Buffer Subtracted Vector
    } opcode_e;

    // ---------------------------------------------------------
    // 2. MICRO-OPERATION CONTROL WORD
    // ---------------------------------------------------------
    // This struct maps 1:1 with the control ports on the datapath
    typedef struct packed {
        // TR Array Controls
        logic       tr_lane0_mode;
        logic [1:0] tr_shift_mode;
        logic       tr_exp_sel;
        
        // Routing MUX Controls
        logic       mux_sub_val_sel;
        logic [1:0] mux_tr_vec_sel;
        logic [1:0] mux_a_sel;
        logic [2:0] mux_b_sel;
        
        // MAC Engine Controls
        logic [1:0] mac_op_mode;
        logic       mac_elemwise;
        logic       mac_clear_acc;
        
        // State Captures & Special Modes
        logic       save_sum;   // Latches raw out_dot (SoftMax Denom OR LN Variance)
        logic       save_mean;  // Latches shifted out_dot (LN Mean)
        logic       gelu_mode;
    } micro_op_t;

    // A convenient "Zero" state for idle/reset
    parameter micro_op_t MICRO_OP_IDLE = '{default: '0};

endpackage