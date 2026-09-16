`timescale 1ns/1ps

/*
 * @module   tr_soc_ctrl_int
 * @brief    TODO: Add one-line description
 * @details  TODO: Add detailed description
 *
 * @param    N               TODO: Add description
 * @param    W               TODO: Add description
 * @param    ACC_W           TODO: Add description
 */
module tr_soc_ctrl_int #(
    parameter int N     = 8,
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic clk,
    input  logic rst_n,

    // MMIO Interface
    input  logic [7:0]              mmio_addr,
    input  logic [31:0]             mmio_wdata,
    input  logic mmio_wen,
    output logic [31:0]             mmio_rdata,

    // Requantizer Parameters
    output logic [ACC_W-1:0]             req_mult_out,
    output logic [5:0]              req_shift_out,

    // VPU Crossbar Routing Controls
    output logic [2:0]              mux_bb_in_sel,
    output logic [1:0]              mux_mac_a_sel,
    output logic [1:0]              mux_mac_b_sel,
    output logic [1:0]              mux_vecmul_a_sel,
    output logic [1:0]              mux_vecmul_b_sel,
    output logic [2:0]              mux_vpu_out_sel,
    output logic en_piped_max,
    output logic en_mac_valid,
    output logic en_vecmul_valid,
    output logic en_bb_valid,
    output logic mac_clear_acc,
    output logic [1:0]              mac_op_mode,
    output logic [1:0]              vecmul_op_mode,
    output logic [1:0]              vecmul_scale_mode,
    output logic bb_shift_mode,
    output logic bb_bypass_ln,
    output logic bb_mode_pre_ln,
    output logic [1:0]              bb_mode_post_ln,
    output logic sym_mode_en,
    output logic signed [W-1:0]     ctrl_scalar_sub_val,

    // Datapath Routing
    output logic src_sram_a_sel, 
    output logic src_sram_b_sel, 
    output logic write_ext_sram, 
    output logic signed [W-1:0]     scratch_a_out [N],
    output logic signed [W-1:0]     scratch_b_out [N],
    
    // VPU Status Inputs (Split Pipeline Valids)
    input  logic signed [W-1:0]     vpu_data_out [N],
    input  logic signed [W-1:0]     vpu_max_out,
    input  logic signed [ACC_W-1:0] vpu_dot_out,
    input  logic vpu_bb_valid,
    input  logic vpu_vecmul_valid,
    input  logic vpu_mac_valid,
    input  logic vpu_max_valid,

    // Streaming matmul dispatch (CMD=0x04) — drives tr_matmul_ctrl in the top.
    output logic         mm_start,
    output logic [15:0]  mm_num_row_tiles,
    output logic [15:0]  mm_num_ctiles,
    input  logic         mm_done
);

    localparam logic [N-1:0]          ADDR_CMD       = 8'h00;
    localparam logic [N-1:0]          ADDR_STATUS    = 8'h04;
    localparam logic [N-1:0]          ADDR_REQ_MULT  = 8'h08;
    localparam logic [N-1:0]          ADDR_REQ_SHIFT = 8'h0C;
    localparam logic [N-1:0]          ADDR_MM_ROWS   = 8'h10;  // matmul: output row tiles
    localparam logic [N-1:0]          ADDR_MM_CTILES = 8'h14;  // matmul: contraction tiles
    localparam logic signed [W-1:0] CONST_LN_SQRT_N = 8'd17;
    // Saturation bounds for the reg_scalar_max + reg_scalar_log add below --
    // each operand fits in W bits, but their sum needs W+1 to avoid wrapping.
    localparam logic signed [W:0]   SUB_VAL_MAX = (1 <<< (W-1)) - 1;
    localparam logic signed [W:0]   SUB_VAL_MIN = -(1 <<< (W-1));

    logic [ACC_W-1:0]                    reg_req_mult;
    logic [5:0]                     reg_req_shift;
    logic [15:0]                    reg_mm_rows;
    logic [15:0]                    reg_mm_ctiles;
    logic                           reg_busy;
    logic                           reg_done;
    logic [N-1:0]                     cmd_trigger;

    logic signed [W-1:0]            scratch_a [N];
    logic signed [W-1:0]            scratch_b [N];
    logic signed [W-1:0]            reg_scalar_max;
    logic signed [W-1:0]            reg_scalar_log;
    
    logic                           write_scratch_a;
    logic                           write_scratch_b;
    logic                           latch_max;
    logic                           latch_log;

    // RMSNorm Pass 3 sign-guard fix (see the RM_P3 case block below for the
    // full derivation): true only during RM_P3/RM_P3_W/RM_P3_MUL_W when
    // ctrl_scalar (reg_scalar_log + CONST_LN_SQRT_N) came out positive --
    // defaults to 0 everywhere else (including GELU's GL_P2, which also
    // uses write_scratch_b), so this cannot affect any other sequence.
    logic                           rm_offset_was_positive;

    assign req_mult_out     = reg_req_mult;
    assign req_shift_out    = reg_req_shift;
    assign scratch_a_out    = scratch_a;
    assign scratch_b_out    = scratch_b;
    assign mm_num_row_tiles = reg_mm_rows;
    assign mm_num_ctiles    = reg_mm_ctiles;

    always_comb begin
        mmio_rdata = '0;
        if (mmio_addr == ADDR_STATUS)    mmio_rdata = {30'd0, reg_done, reg_busy};
        if (mmio_addr == ADDR_REQ_MULT)  mmio_rdata = reg_req_mult;
        if (mmio_addr == ADDR_REQ_SHIFT) mmio_rdata = {26'd0, reg_req_shift};
    end

    // RMSNorm Pass 3 sign-guard fix, continued: reciprocal LUT. E (the
    // decay-side exp result the guarded path always produces now)
    // realistically ranges 0..16 (Q4.4 "1.0" = 16, is_zero case); sized to
    // 0..31 for headroom. recip_lut[E] = round(256/E), saturated to 127
    // (max signed 8-bit) -- 256 = 16*16, converting Q4.4 E back out through
    // the same Q4.4 convention this datapath already uses everywhere else.
    // Root cause + fix verified numerically in the bit-true emulation model
    // (tr_math_model.hpp's rmsnorm_hw_model_signguard_fixed()) and in RTL
    // simulation against tr_rmsnorm.sv (see docs/iscas_paper_support/);
    // this is that same fix applied to the production FSM/VPU path. New
    // logic scoped entirely to this module -- tr_nonlinear_vpu (the shared
    // backbone GELU and Softmax also use) is untouched.
    function automatic logic signed [W-1:0] recip_lut(input logic signed [W-1:0] e);
        case (e)
            8'sd0,  8'sd1: recip_lut = 8'sd127;
            8'sd2:         recip_lut = 8'sd127;
            8'sd3:         recip_lut = 8'sd85;
            8'sd4:         recip_lut = 8'sd64;
            8'sd5:         recip_lut = 8'sd51;
            8'sd6:         recip_lut = 8'sd43;
            8'sd7:         recip_lut = 8'sd37;
            8'sd8:         recip_lut = 8'sd32;
            8'sd9:         recip_lut = 8'sd28;
            8'sd10:        recip_lut = 8'sd26;
            8'sd11:        recip_lut = 8'sd23;
            8'sd12:        recip_lut = 8'sd21;
            8'sd13:        recip_lut = 8'sd20;
            8'sd14:        recip_lut = 8'sd18;
            8'sd15:        recip_lut = 8'sd17;
            8'sd16:        recip_lut = 8'sd16;
            8'sd17:        recip_lut = 8'sd15;
            8'sd18:        recip_lut = 8'sd14;
            8'sd19:        recip_lut = 8'sd13;
            8'sd20:        recip_lut = 8'sd13;
            8'sd21:        recip_lut = 8'sd12;
            8'sd22:        recip_lut = 8'sd12;
            8'sd23:        recip_lut = 8'sd11;
            8'sd24:        recip_lut = 8'sd11;
            8'sd25:        recip_lut = 8'sd10;
            8'sd26:        recip_lut = 8'sd10;
            8'sd27:        recip_lut = 8'sd9;
            8'sd28:        recip_lut = 8'sd9;
            8'sd29:        recip_lut = 8'sd9;
            8'sd30:        recip_lut = 8'sd9;
            8'sd31:        recip_lut = 8'sd8;
            default:       recip_lut = 8'sd127;   // E outside the expected range -- saturate, don't wrap
        endcase
    endfunction

    always_ff @(posedge clk) begin
        if (write_scratch_a) scratch_a <= vpu_data_out;
        if (write_scratch_b) begin
            if (rm_offset_was_positive) begin
                for (int k = 0; k < N; k++) scratch_b[k] <= recip_lut(vpu_data_out[k]);
            end else begin
                scratch_b <= vpu_data_out;
            end
        end
        if (latch_max)       reg_scalar_max <= vpu_max_out;
        if (latch_log)       reg_scalar_log <= vpu_data_out[0];
    end

    // Expanded FSM to handle chained pipelining (Backbone -> VecMul/MAC)
    typedef enum logic [5:0] {
        IDLE,
        // Streaming matmul (CMD=0x04)
        MM_START, MM_WAIT,
        // GELU
        GL_P1, GL_P1_W, GL_P1_MUL, GL_P1_MUL_W,
        GL_P2, GL_P2_W, GL_P2_MUL, GL_P2_MUL_W,
        GL_P3, GL_P3_W,
        // Softmax
        SM_P1, SM_P1_W,
        SM_P2, SM_P2_W1, SM_P2_MAC, SM_P2_W2,
        SM_P3, SM_P3_W,
        SM_P4, SM_P4_W, SM_P4_MUL, SM_P4_MUL_W,
        // RMSNorm
        RM_P1, RM_P1_W,
        RM_P2, RM_P2_W,
        RM_P3, RM_P3_W, RM_P3_MUL, RM_P3_MUL_W,
        RM_P4, RM_P4_W,
        DONE
    } state_t;
    
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            reg_req_mult  <= '0;
            reg_req_shift <= '0;
            reg_mm_rows   <= '0;
            reg_mm_ctiles <= '0;
            cmd_trigger   <= '0;
        end else begin
            state         <= next_state;
            cmd_trigger   <= 8'h00;

            if (mmio_wen) begin
                if (mmio_addr == ADDR_CMD)       cmd_trigger   <= mmio_wdata[7:0];
                if (mmio_addr == ADDR_REQ_MULT)  reg_req_mult  <= mmio_wdata;
                if (mmio_addr == ADDR_REQ_SHIFT) reg_req_shift <= mmio_wdata[5:0];
                if (mmio_addr == ADDR_MM_ROWS)   reg_mm_rows   <= mmio_wdata[15:0];
                if (mmio_addr == ADDR_MM_CTILES) reg_mm_ctiles <= mmio_wdata[15:0];
            end
        end
    end

    always_comb begin
        // Safe Global Defaults
        next_state          = state;
        reg_busy            = 1'b1;
        reg_done            = 1'b0;
        mm_start            = 1'b0;
        
        write_ext_sram      = 1'b0;
        write_scratch_a     = 1'b0;
        write_scratch_b     = 1'b0;
        latch_max           = 1'b0;
        latch_log           = 1'b0;
        
        src_sram_a_sel      = 1'b0;
        src_sram_b_sel      = 1'b0;
        
        mux_bb_in_sel       = 3'b011;
        mux_mac_a_sel       = 2'b00;
        mux_mac_b_sel       = 2'b00;
        mux_vecmul_a_sel    = 2'b00;
        mux_vecmul_b_sel    = 2'b00;
        mux_vpu_out_sel     = 3'b000;
        
        en_piped_max        = 1'b0;
        en_mac_valid        = 1'b0;
        en_vecmul_valid     = 1'b0;
        en_bb_valid         = 1'b0;
        
        mac_clear_acc       = 1'b0;
        mac_op_mode         = 2'd0;
        vecmul_op_mode      = 2'd0;
        vecmul_scale_mode   = 2'b00;
        bb_shift_mode       = 1'b0;
        
        bb_bypass_ln        = 1'b0;
        bb_mode_pre_ln      = 1'b0;
        bb_mode_post_ln     = 2'b00;
        sym_mode_en         = 1'b0;
        ctrl_scalar_sub_val = '0;
        rm_offset_was_positive = 1'b0;

        case (state)
            IDLE: begin
                reg_busy = 1'b0;
                if (cmd_trigger == 8'h01) next_state = SM_P1;
                if (cmd_trigger == 8'h02) next_state = GL_P1;
                if (cmd_trigger == 8'h03) next_state = RM_P1;
                if (cmd_trigger == 8'h04) next_state = MM_START;
            end

            // =========================================================
            // STREAMING MATMUL: kick tr_matmul_ctrl and wait for it.
            // =========================================================
            MM_START: begin
                mm_start   = 1'b1;
                next_state = MM_WAIT;
            end
            MM_WAIT: begin
                if (mm_done) next_state = DONE;
            end

            // =========================================================
            // GELU SEQUENCE (Grouped States prevent combinational drift)
            // =========================================================
            GL_P1, GL_P1_W, GL_P1_MUL_W: begin
                bb_bypass_ln = 1'b1; 
                mux_bb_in_sel     = 3'b001;
                vecmul_op_mode    = 2'd2;       // UU Mode for Exponential!
                vecmul_scale_mode = 2'b10;      // Slice [15:8]
                mux_vecmul_a_sel  = 2'b11;
                mux_vecmul_b_sel  = 2'b10;
                mux_vpu_out_sel   = 3'b000;

                if (state == GL_P1) begin 
                    en_bb_valid = 1'b1;
                    next_state  = GL_P1_W; 
                end else if (state == GL_P1_W && vpu_bb_valid) begin 
                    en_vecmul_valid = 1'b1;
                    next_state      = GL_P1_MUL_W; 
                end else if (state == GL_P1_MUL_W && vpu_vecmul_valid) begin 
                    write_scratch_a = 1'b1;
                    next_state      = GL_P2; 
                end
            end

            GL_P2, GL_P2_W, GL_P2_MUL_W: begin
                src_sram_a_sel    = 1'b1;
                mux_bb_in_sel     = 3'b011; 
                bb_mode_pre_ln    = 1'b1;
                bb_mode_post_ln   = 2'b01;
                vecmul_op_mode    = 2'd2; 
                vecmul_scale_mode = 2'b10;
                mux_vecmul_a_sel  = 2'b11;
                mux_vecmul_b_sel  = 2'b10;
                mux_vpu_out_sel   = 3'b000;

                if (state == GL_P2) begin 
                    en_bb_valid = 1'b1;
                    next_state  = GL_P2_W; 
                end else if (state == GL_P2_W && vpu_bb_valid) begin 
                    en_vecmul_valid = 1'b1;
                    next_state      = GL_P2_MUL_W; 
                end else if (state == GL_P2_MUL_W && vpu_vecmul_valid) begin 
                    write_scratch_b = 1'b1;
                    next_state      = GL_P3; 
                end
            end

            GL_P3, GL_P3_W: begin
                src_sram_a_sel    = 1'b0;
                src_sram_b_sel    = 1'b1;
                sym_mode_en       = 1'b1; 
                vecmul_op_mode    = 2'd1;       // SU Mode
                vecmul_scale_mode = 2'b01;      // Slice [11:4]
                mux_vecmul_a_sel  = 2'b00;
                mux_vecmul_b_sel  = 2'b01;
                mux_vpu_out_sel   = 3'b000;

                if (state == GL_P3) begin 
                    en_vecmul_valid = 1'b1;
                    next_state      = GL_P3_W; 
                end else if (state == GL_P3_W && vpu_vecmul_valid) begin 
                    write_ext_sram = 1'b1;
                    next_state     = DONE; 
                end
            end

            // =========================================================
            // SOFTMAX SEQUENCE (Fixed Pipeline Latency)
            // =========================================================
            SM_P1, SM_P1_W: begin
                en_piped_max = 1;
                // Wait for the piped_max tree's own valid handshake before latching
                // (its comparison tree takes $clog2(N) cycles to settle).
                if (state == SM_P1) next_state = SM_P1_W;
                else if (state == SM_P1_W && vpu_max_valid) begin latch_max = 1; next_state = SM_P2; end
            end
            
            SM_P2, SM_P2_W1, SM_P2_W2: begin
                bb_bypass_ln = 1'b1;
                ctrl_scalar_sub_val = reg_scalar_max; mux_bb_in_sel = 3'b010; 
                mac_op_mode = 2'd2; // UU
                mux_mac_a_sel = 2'b01; mux_mac_b_sel = 2'b01; // e_a * mantisa
                
                if (state == SM_P2) begin en_bb_valid = 1; next_state = SM_P2_W1; end
                else if (state == SM_P2_W1 && vpu_bb_valid) begin mac_clear_acc = 1; en_mac_valid = 1; next_state = SM_P2_W2; end
                else if (state == SM_P2_W2 && vpu_mac_valid) next_state = SM_P3; 
            end
            
            SM_P3, SM_P3_W: begin
                bb_shift_mode = 1'b0; // Use [23:8]
                mux_bb_in_sel = 3'b000; mux_vpu_out_sel = 3'b001; 
                if (state == SM_P3) begin en_bb_valid = 1; next_state = SM_P3_W; end
                else if (state == SM_P3_W && vpu_bb_valid) begin latch_log = 1; next_state = SM_P4; end
            end
            
            SM_P4, SM_P4_W, SM_P4_MUL_W: begin
                logic signed [W:0] sub_val_wide;
                bb_bypass_ln = 1'b1;
                // Saturating add (was a raw W-bit add that could wrap: e.g.
                // max=125 + log_sum=3 = 128 silently became -128, corrupting
                // the exponent argument for every lane in the row). max and
                // log_sum are each valid Q4.4 int8 values individually, but
                // their sum can exceed the W-bit range, so widen before
                // clamping instead of truncating in the assignment.
                sub_val_wide = $signed({reg_scalar_max[W-1], reg_scalar_max}) +
                               $signed({reg_scalar_log[W-1], reg_scalar_log});
                if (sub_val_wide > SUB_VAL_MAX)      ctrl_scalar_sub_val = SUB_VAL_MAX[W-1:0];
                else if (sub_val_wide < SUB_VAL_MIN) ctrl_scalar_sub_val = SUB_VAL_MIN[W-1:0];
                else                                  ctrl_scalar_sub_val = sub_val_wide[W-1:0];
                mux_bb_in_sel = 3'b010;
                vecmul_op_mode = 2'd2; vecmul_scale_mode = 2'b11; // UU, Q0.8 unsigned (saturating)
                mux_vecmul_a_sel = 2'b11; mux_vecmul_b_sel = 2'b10;
                mux_vpu_out_sel = 3'b000;

                if (state == SM_P4) begin en_bb_valid = 1; next_state = SM_P4_W; end
                else if (state == SM_P4_W && vpu_bb_valid) begin en_vecmul_valid = 1; next_state = SM_P4_MUL_W; end
                else if (state == SM_P4_MUL_W && vpu_vecmul_valid) begin write_ext_sram = 1; next_state = DONE; end
            end

            // =========================================================
            // RMSNORM SEQUENCE (Fixed 4-Pass Math Route)
            // =========================================================
            RM_P1, RM_P1_W: begin
                mux_mac_a_sel = 2'b00; mux_mac_b_sel = 2'b10; // Route X * X
                mac_op_mode = 2'd0; // SS
                if (state == RM_P1) begin mac_clear_acc = 1; en_mac_valid = 1; next_state = RM_P1_W; end
                else if (state == RM_P1_W && vpu_mac_valid) next_state = RM_P2;
            end
            
            RM_P2, RM_P2_W: begin
                bb_shift_mode = 1'b1; // Use [19:4]
                mux_bb_in_sel = 3'b000; bb_mode_post_ln = 2'b10; // -0.5x
                mux_vpu_out_sel = 3'b001; 
                if (state == RM_P2) begin en_bb_valid = 1; next_state = RM_P2_W; end
                else if (state == RM_P2_W && vpu_bb_valid) begin latch_log = 1; next_state = RM_P3; end
            end
            
            // NEW PASS 3: Reconstruct InvRMS and save to Scratch B
            RM_P3, RM_P3_W, RM_P3_MUL_W: begin
                logic signed [W-1:0] rm_ctrl_scalar_raw;
                bb_bypass_ln = 1'b1;

                // SIGN-GUARD FIX: tr_nonlinear_vpu's shared exp backbone
                // (round.sv's own comment: "8-bit Negative-Only LUT
                // Indexing") was only ever built for alpha_stabilizer-style
                // inputs, which are always forced non-positive.
                // rm_ctrl_scalar_raw can legitimately be positive for small
                // Sum(x^2) -- a normal, expected input, not a corner case --
                // and feeding that straight to the backbone silently
                // collapses InvRMS toward 0 instead of growing it, as it
                // mathematically should (root cause verified in
                // docs/iscas_paper_support/). Force it non-positive before
                // it reaches the backbone (cheap: comparator + negate,
                // mirrors alpha_stabilizer's own "forced negative absolute
                // value" step) and remember that a flip happened via
                // rm_offset_was_positive; the value is stable for the whole
                // RM_P3/RM_P3_W/RM_P3_MUL_W wait sequence since
                // reg_scalar_log doesn't change during it, so no extra
                // pipelining is needed -- the flag is read back at the
                // write_scratch_b capture point above, in the very same
                // cycle it's set here (RM_P3_MUL_W). The flip itself is
                // undone there (see the recip_lut() call), not here.
                rm_ctrl_scalar_raw = reg_scalar_log + CONST_LN_SQRT_N;
                rm_offset_was_positive = ~rm_ctrl_scalar_raw[W-1] && (rm_ctrl_scalar_raw != '0);
                ctrl_scalar_sub_val = rm_offset_was_positive ? -rm_ctrl_scalar_raw : rm_ctrl_scalar_raw;

                src_sram_a_sel = 1; // Force 0 by reading empty Scratch A
                mux_bb_in_sel = 3'b100; 
                vecmul_op_mode = 2'd2; vecmul_scale_mode = 2'b10; // UU [15:8]
                mux_vecmul_a_sel = 2'b11; mux_vecmul_b_sel = 2'b10;
                mux_vpu_out_sel = 3'b000;

                if (state == RM_P3) begin en_bb_valid = 1; next_state = RM_P3_W; end
                else if (state == RM_P3_W && vpu_bb_valid) begin en_vecmul_valid = 1; next_state = RM_P3_MUL_W; end
                else if (state == RM_P3_MUL_W && vpu_vecmul_valid) begin write_scratch_b = 1; next_state = RM_P4; end
            end

            // NEW PASS 4: Multiply X * Scratch B (InvRMS)
            RM_P4, RM_P4_W: begin
                src_sram_a_sel = 0; src_sram_b_sel = 1; 
                vecmul_op_mode = 2'd1; // SU (Signed X * Unsigned InvRMS)
                vecmul_scale_mode = 2'b01; // Slice [11:4] for Q4.4 output
                mux_vecmul_a_sel = 2'b00; mux_vecmul_b_sel = 2'b00;
                mux_vpu_out_sel = 3'b000;

                if (state == RM_P4) begin en_vecmul_valid = 1; next_state = RM_P4_W; end
                else if (state == RM_P4_W && vpu_vecmul_valid) begin write_ext_sram = 1; next_state = DONE; end
            end

            DONE: begin
                reg_done   = 1'b1;
                reg_busy   = 1'b0;
                next_state = IDLE;
            end
        endcase
    end
endmodule