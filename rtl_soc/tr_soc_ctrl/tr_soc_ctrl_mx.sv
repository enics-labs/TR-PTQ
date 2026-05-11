`timescale 1ns/1ps

module tr_soc_ctrl_mx #(
    parameter int N     = 8,
    parameter int W     = 8,
    parameter int ACC_W = 32
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // MMIO Interface
    input  logic [7:0]              mmio_addr,
    input  logic [31:0]             mmio_wdata,
    input  logic                    mmio_wen,
    output logic [31:0]             mmio_rdata,

    // MX Dynamic Shifter Control
    output logic                    ctrl_enable_linear_shift,

    // VPU Crossbar Routing Controls 
    output logic [2:0]              mux_bb_in_sel,
    output logic [1:0]              mux_mac_a_sel,
    output logic [1:0]              mux_mac_b_sel,
    output logic [1:0]              mux_vecmul_a_sel,
    output logic [1:0]              mux_vecmul_b_sel,
    output logic [2:0]              mux_vpu_out_sel,

    output logic                    en_piped_max,
    output logic                    en_mac_valid,
    output logic                    en_vecmul_valid,
    output logic                    en_bb_valid,
    
    output logic                    mac_clear_acc,
    output logic [1:0]              mac_op_mode,
    output logic [1:0]              vecmul_op_mode,
    output logic [1:0]              vecmul_scale_mode,
    
    output logic                    bb_shift_mode,
    output logic                    bb_bypass_ln,
    output logic                    bb_mode_pre_ln,
    output logic [1:0]              bb_mode_post_ln,
    output logic                    sym_mode_en,
    output logic signed [W-1:0]     ctrl_scalar_sub_val,

    // Datapath Routing
    output logic                    src_sram_a_sel, 
    output logic                    src_sram_b_sel, 
    output logic                    write_ext_sram, 
    output logic signed [W-1:0]     scratch_a_out [N],
    output logic signed [W-1:0]     scratch_b_out [N],
    
    // VPU Status Inputs
    input  logic signed [W-1:0]     vpu_data_out [N],
    input  logic signed [W-1:0]     vpu_max_out,
    input  logic signed [ACC_W-1:0] vpu_dot_out,
    input  logic                    vpu_bb_valid,
    input  logic                    vpu_vecmul_valid,
    input  logic                    vpu_mac_valid
);

    localparam logic [7:0]          ADDR_CMD       = 8'h00;
    localparam logic [7:0]          ADDR_STATUS    = 8'h04;
    localparam logic signed [W-1:0] CONST_LN_SQRT_N = 8'd17; 

    logic                           reg_busy;
    logic                           reg_done;
    logic [7:0]                     cmd_trigger;
    logic signed [W-1:0]            scratch_a [N];
    logic signed [W-1:0]            scratch_b [N];
    logic signed [W-1:0]            reg_scalar_max;
    logic signed [W-1:0]            reg_scalar_log;
    
    logic                           write_scratch_a;
    logic                           write_scratch_b;
    logic                           latch_max;
    logic                           latch_log;

    assign scratch_a_out = scratch_a;
    assign scratch_b_out = scratch_b;

    always_comb begin
        mmio_rdata = '0;
        if (mmio_addr == ADDR_STATUS) mmio_rdata = {30'd0, reg_done, reg_busy};
    end

    always_ff @(posedge clk) begin
        if (write_scratch_a) scratch_a <= vpu_data_out;
        if (write_scratch_b) scratch_b <= vpu_data_out;
        if (latch_max)       reg_scalar_max <= vpu_max_out;
        if (latch_log)       reg_scalar_log <= vpu_data_out[0];
    end

    typedef enum logic [5:0] {
        IDLE,
        GL_P1, GL_P1_W, GL_P1_MUL, GL_P1_MUL_W,
        GL_P2, GL_P2_W, GL_P2_MUL, GL_P2_MUL_W,
        GL_P3, GL_P3_W,
        SM_P1, SM_P1_W1, SM_P1_W2, SM_P1_W,
        SM_P2, SM_P2_W1, SM_P2_MAC, SM_P2_W2,
        SM_P3, SM_P3_W,
        SM_P4, SM_P4_W, SM_P4_MUL, SM_P4_MUL_W,
        RM_P1, RM_P1_W,
        RM_P2, RM_P2_W,
        RM_P3, RM_P3_W, RM_P3_MUL, RM_P3_MUL_W,
        RM_P4, RM_P4_W,
        DONE
    } state_t;
    
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            cmd_trigger <= '0;
        end else begin
            state       <= next_state;
            cmd_trigger <= 8'h00; 
            if (mmio_wen && mmio_addr == ADDR_CMD) begin
                cmd_trigger <= mmio_wdata[7:0];
            end
        end
    end

    always_comb begin
        next_state               = state;
        reg_busy                 = 1'b1;
        reg_done                 = 1'b0;
        ctrl_enable_linear_shift = 1'b0; 
        
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

        case (state)
            IDLE: begin
                reg_busy = 1'b0;
                if (cmd_trigger == 8'h01) next_state = SM_P1;
                if (cmd_trigger == 8'h02) next_state = GL_P1;
                if (cmd_trigger == 8'h03) next_state = RM_P1;
            end

            GL_P1, GL_P1_W, GL_P1_MUL_W: begin
                ctrl_enable_linear_shift = 1'b1;
                bb_bypass_ln = 1'b1;
                mux_bb_in_sel = 3'b001;
                vecmul_op_mode = 2'd2;
                vecmul_scale_mode = 2'b10;
                mux_vecmul_a_sel = 2'b11;
                mux_vecmul_b_sel = 2'b10;
                
                if (state == GL_P1) begin en_bb_valid = 1; next_state = GL_P1_W; end
                else if (state == GL_P1_W && vpu_bb_valid) begin en_vecmul_valid = 1; next_state = GL_P1_MUL_W; end
                else if (state == GL_P1_MUL_W && vpu_vecmul_valid) begin write_scratch_a = 1; next_state = GL_P2; end
            end

            GL_P2, GL_P2_W, GL_P2_MUL_W: begin
                ctrl_enable_linear_shift = 1'b1;
                src_sram_a_sel = 1'b1;
                mux_bb_in_sel = 3'b011; 
                bb_mode_pre_ln = 1'b1;
                bb_mode_post_ln = 2'b01;
                vecmul_op_mode = 2'd2; 
                vecmul_scale_mode = 2'b10;
                mux_vecmul_a_sel = 2'b11;
                mux_vecmul_b_sel = 2'b10;

                if (state == GL_P2) begin en_bb_valid = 1; next_state = GL_P2_W; end
                else if (state == GL_P2_W && vpu_bb_valid) begin en_vecmul_valid = 1; next_state = GL_P2_MUL_W; end
                else if (state == GL_P2_MUL_W && vpu_vecmul_valid) begin write_scratch_b = 1; next_state = GL_P3; end
            end

            GL_P3, GL_P3_W: begin
                ctrl_enable_linear_shift = 1'b1;
                src_sram_a_sel = 1'b0;
                src_sram_b_sel = 1'b1;
                sym_mode_en = 1'b1;
                vecmul_op_mode = 2'd1;
                vecmul_scale_mode = 2'b01;
                mux_vecmul_a_sel = 2'b00;
                mux_vecmul_b_sel = 2'b01;

                if (state == GL_P3) begin en_vecmul_valid = 1; next_state = GL_P3_W; end
                else if (state == GL_P3_W && vpu_vecmul_valid) begin write_ext_sram = 1; next_state = DONE; end
            end

            SM_P1, SM_P1_W1, SM_P1_W2, SM_P1_W: begin
                ctrl_enable_linear_shift = 1'b1;
                en_piped_max = 1;
                if (state == SM_P1) next_state = SM_P1_W1;
                else if (state == SM_P1_W1) next_state = SM_P1_W2;
                else if (state == SM_P1_W2) next_state = SM_P1_W;
                else if (state == SM_P1_W) begin latch_max = 1; next_state = SM_P2; end
            end
            
            SM_P2, SM_P2_W1, SM_P2_W2: begin
                ctrl_enable_linear_shift = 1'b1;
                bb_bypass_ln = 1'b1;
                ctrl_scalar_sub_val = reg_scalar_max; mux_bb_in_sel = 3'b010; 
                mac_op_mode = 2'd2;
                mux_mac_a_sel = 2'b01;
                mux_mac_b_sel = 2'b01;
                
                if (state == SM_P2) begin en_bb_valid = 1; next_state = SM_P2_W1; end
                else if (state == SM_P2_W1 && vpu_bb_valid) begin mac_clear_acc = 1; en_mac_valid = 1; next_state = SM_P2_W2; end
                else if (state == SM_P2_W2 && vpu_mac_valid) next_state = SM_P3;
            end
            
            SM_P3, SM_P3_W: begin
                ctrl_enable_linear_shift = 1'b1;
                bb_shift_mode = 1'b0;
                mux_bb_in_sel = 3'b000;
                mux_vpu_out_sel = 3'b001; 
                if (state == SM_P3) begin en_bb_valid = 1; next_state = SM_P3_W; end
                else if (state == SM_P3_W && vpu_bb_valid) begin latch_log = 1; next_state = SM_P4; end
            end
            
            SM_P4, SM_P4_W, SM_P4_MUL_W: begin
                ctrl_enable_linear_shift = 1'b1;
                bb_bypass_ln = 1'b1;
                ctrl_scalar_sub_val = reg_scalar_max + reg_scalar_log;
                mux_bb_in_sel = 3'b010; 
                vecmul_op_mode = 2'd2; vecmul_scale_mode = 2'b10;
                mux_vecmul_a_sel = 2'b11;
                mux_vecmul_b_sel = 2'b10;

                if (state == SM_P4) begin en_bb_valid = 1; next_state = SM_P4_W; end
                else if (state == SM_P4_W && vpu_bb_valid) begin en_vecmul_valid = 1; next_state = SM_P4_MUL_W; end
                else if (state == SM_P4_MUL_W && vpu_vecmul_valid) begin write_ext_sram = 1; next_state = DONE; end
            end

            RM_P1, RM_P1_W: begin
                mux_mac_a_sel = 2'b00;
                mux_mac_b_sel = 2'b10;
                mac_op_mode = 2'd0;
                if (state == RM_P1) begin mac_clear_acc = 1; en_mac_valid = 1; next_state = RM_P1_W; end
                else if (state == RM_P1_W && vpu_mac_valid) next_state = RM_P2;
            end
            
            RM_P2, RM_P2_W: begin
                bb_shift_mode = 1'b1;
                mux_bb_in_sel = 3'b000;
                bb_mode_post_ln = 2'b10;
                mux_vpu_out_sel = 3'b001;
                if (state == RM_P2) begin en_bb_valid = 1; next_state = RM_P2_W; end
                else if (state == RM_P2_W && vpu_bb_valid) begin latch_log = 1; next_state = RM_P3; end
            end
            
            RM_P3, RM_P3_W, RM_P3_MUL_W: begin
                bb_bypass_ln = 1'b1;
                ctrl_scalar_sub_val = reg_scalar_log + CONST_LN_SQRT_N;
                src_sram_a_sel = 1;
                mux_bb_in_sel = 3'b100;
                vecmul_op_mode = 2'd2; vecmul_scale_mode = 2'b10;
                mux_vecmul_a_sel = 2'b11;
                mux_vecmul_b_sel = 2'b10;

                if (state == RM_P3) begin en_bb_valid = 1; next_state = RM_P3_W; end
                else if (state == RM_P3_W && vpu_bb_valid) begin en_vecmul_valid = 1; next_state = RM_P3_MUL_W; end
                else if (state == RM_P3_MUL_W && vpu_vecmul_valid) begin write_scratch_b = 1; next_state = RM_P4; end
            end

            RM_P4, RM_P4_W: begin
                src_sram_a_sel = 0;
                src_sram_b_sel = 1; 
                vecmul_op_mode = 2'd1;
                vecmul_scale_mode = 2'b01;
                mux_vecmul_a_sel = 2'b00;
                mux_vecmul_b_sel = 2'b00;

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