`timescale 1ns/1ps

/*
 * @module   tr_soc_ctrl_int
 * @brief    MMIO-driven master FSM controller for the INT-datapath SoC: on
 *           an MMIO command write, sequences tr_nonlinear_vpu's crossbar
 *           through the multi-pass Softmax/RMSNorm/SwiGLU routes, or kicks off
 *           a streaming matmul via tr_matmul_ctrl.
 * @details  MMIO register map: ADDR_CMD (0x00, write triggers a command:
 *           1=Softmax, 4=streaming matmul, 5=SiLU group (SwiGLU),
 *           6=SwiGLU max-tracking group, 0xB=SwiGLU row gain, 0xC=SwiGLU apply group,
 *           7/8=RMSNorm sum-of-squares beat (7: first tile, clears the accumulator; 8: next tile,
 *           accumulates), 9=RMSNorm scale (once per row: mean -> ln -> exp -> wide inv_rms register),
 *           0xA=RMSNorm apply (per tile: x * inv_rms)),
 *           ADDR_STATUS (0x04, {done,busy}), ADDR_REQ_MULT/ADDR_REQ_SHIFT
 *           (0x08/0x0C, requantizer M/S), ADDR_MM_ROWS/ADDR_MM_CTILES
 *           (0x10/0x14, streaming-matmul tile counts for tr_matmul_ctrl),
 *           ADDR_RMS_LOG2N (0x18, log2 of the RMSNorm row length, read by command 9),
 *           ADDR_SW_E (0x1C, read-only: SwiGLU row gain e, signed),
 *           ADDR_SM_Z (0x20, read-only: softmax tile z = sat8(max + ln(S)), valid after command 1).
 *
 *           Each command is a fixed sequence of states (SM_*, RM_*, SW_*) driving tr_nonlinear_vpu's crossbar-select/enable/mode
 *           outputs pass-by-pass, gated on that pass's own vpu_*_valid
 *           handshake (the VPU's submodules have no external ready signal,
 *           so each pass waits out its producer's own valid pulse before
 *           advancing). scratch_a holds the SwiGLU silu
 *           intermediate between passes; reg_scalar_max/reg_scalar_log hold
 *           Softmax's row-max and log-sum-exp scalars, and RMSNorm's
 *           log(sum-of-squares) scalar, between their own passes.
 *
 *           SwiGLU (ffn_h = silu(gate) * up, rows of 8*groups elements) is host-sequenced in 8-lane groups,
 *           two passes per row with a per-row output gain (tools/models/vpu_ops.py::swiglu, sw_gain="dynamic"):
 *             pass 1, per group: cmd 5 (gate x -> sigma ROM -> silu_s = sat12(round(x*sigma >> 2)) into the wide
 *                                scratch_a), cmd 6 (up u: running max of |silu_s*u| into reg_sw_max)
 *             once per row:      cmd 0xB (gain e from reg_sw_max, clears the max; e readable at ADDR_SW_E)
 *             pass 2, per group: cmd 5 again, cmd 0xC (y = sat8(round_shift(4*silu_s*u, 10-e)) -> ext_sram)
 *           The sigmoid comes from a 256x6-bit sigma ROM in tr_nonlinear_vpu (equal to the exp-ln-exp chain at FRAC_W=6).
 *
 *           RMSNorm is a host-sequenced tile flow (the VPU is N=8 wide): the host issues command 7 for
 *           the first tile of a row and 8 for each further tile (the MAC accumulates S = sum x^2 across
 *           tiles), command 9 once, then command 0xA per tile. Command 9 (RM_P2, RM_P3) turns S into
 *           mk = round(S >> (FRAC_W+log2N-2)), runs the shared ln at BITS=FRAC_W+2, halves with rounding,
 *           and evaluates exp through the sign-guarded backbone; the unsigned wide inv_rms (INV_W bits,
 *           FRAC_W+4 fractional bits) is latched in reg_inv_rms (decay: exp output; growth: 65536/E
 *           from recip_rom in the VPU). Software spec: tools/models/vpu_ops.py::rmsnorm.
 *
 *           RM_P3 includes a sign-guard fix (forces the log-domain argument
 *           non-positive before tr_nonlinear_vpu's shared exp backbone,
 *           which only supports the decay/negative-domain case, then
 *           corrects the result back via recip_lut when the true value was
 *           positive) -- see the inline comment at rm_ctrl_scalar_raw for
 *           the full derivation; this mirrors the same fix in tr_rmsnorm.sv.
 *
 * @param    N      Vector dimension (lanes) of the driven VPU/datapath.
 * @param    W      Data word width (ctrl_scalar_sub_val, vpu_data_out).
 * @param    ACC_W  Accumulator width (req_mult_out, vpu_dot_out).
 */
module tr_soc_ctrl_int #(
    parameter int N     = 8,
    parameter int W     = 8,
    parameter int ACC_W = 32,
    parameter int INV_W = 12,
    parameter int SA_W  = 12,   // width of the SwiGLU silu intermediate (scratch_a)
    parameter int WIDE_W = 20   // width of the VPU wide vecmul result bus
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
    output logic [2:0]              mux_vecmul_b_sel,
    output logic [2:0]              mux_vpu_out_sel,
    output logic en_piped_max,
    output logic en_mac_valid,
    output logic en_vecmul_valid,
    output logic en_bb_valid,
    output logic mac_clear_acc,
    output logic [1:0]              mac_op_mode,
    output logic [1:0]              vecmul_op_mode,
    output logic [2:0]              vecmul_scale_mode,
    output logic bb_shift_mode,
    output logic bb_bypass_ln,
    output logic [1:0]              bb_mode_post_ln,
    output logic bb_ln_pre_scaled,  // 1: ln input already has FRAC_W+2 fractional bits (RMSNorm mean)
    output logic signed [W-1:0]     ctrl_scalar_sub_val,
    output logic        [4:0]       rms_log2n_out,     // RMSNorm row length log2 (ADDR_RMS_LOG2N)
    output logic        [INV_W-1:0] inv_rms_out,       // RMSNorm wide inv_rms (reg_inv_rms)
    output logic signed [5:0]       sw_shift_out,      // SwiGLU apply shift = 10 - e
    output logic                    rms_growth_out,    // RMSNorm: sign-guard flip (selects the reciprocal path in the VPU)
    output logic                    mux_max_sel,       // piped_max input select: 0 softmax x, 1 SwiGLU |prod|

    // Datapath Routing
    output logic src_sram_a_sel, 
    output logic write_ext_sram, 
    output logic signed [SA_W-1:0]  scratch_a_out [N],
    
    // VPU Status Inputs (Split Pipeline Valids)
    input  logic signed [W-1:0]     vpu_data_out [N],
    input  logic signed [WIDE_W-1:0] vpu_wide_out [N],      // full-width vecmul results (SwiGLU)
    input  logic signed [WIDE_W-1:0] vpu_max_out,
    input  logic signed [ACC_W-1:0] vpu_dot_out,
    input  logic        [INV_W-1:0] vpu_inv_rms,      // RMSNorm inv_rms computed by the VPU from the exp result (decay: E, growth: 1/E)
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
    localparam logic [N-1:0]          ADDR_RMS_LOG2N = 8'h18;  // RMSNorm: log2(row length)
    localparam logic [N-1:0]          ADDR_SW_E      = 8'h1C;  // SwiGLU: row gain e (read-only)
    localparam logic [N-1:0]          ADDR_SM_Z      = 8'h20;  // softmax: tile z = sat8(max + ln(S)) (read-only)
    // Saturation bounds for the reg_scalar_max + reg_scalar_log add below --
    // each operand fits in W bits, but their sum needs W+1 to avoid wrapping.
    localparam logic signed [W:0]   SUB_VAL_MAX = (1 <<< (W-1)) - 1;
    localparam logic signed [W:0]   SUB_VAL_MIN = -(1 <<< (W-1));

    logic [ACC_W-1:0]                    reg_req_mult;
    logic [5:0]                     reg_req_shift;
    logic [15:0]                    reg_mm_rows;
    logic [15:0]                    reg_mm_ctiles;
    logic [4:0]                     reg_rms_log2n;
    logic [INV_W-1:0]               reg_inv_rms;
    logic                           latch_inv;
    logic                           reg_busy;
    logic                           reg_done;
    logic [N-1:0]                     cmd_trigger;

    logic signed [SA_W-1:0]         scratch_a [N];
    logic signed [W-1:0]            reg_scalar_max;
    logic signed [W-1:0]            reg_scalar_log;
    logic signed [W-1:0]            sm_z;             // softmax tile z = sat8(max + ln(S)), see below
    
    logic                           write_scratch_a;
    logic                           write_silu;       // SwiGLU: scratch_a <= sat12(round(wide >> 2))
    logic                           latch_sw_max;
    logic                           latch_sw_gain;
    logic [WIDE_W-1:0]              reg_sw_max;       // running max of |silu_s * up| over the row
    logic signed [4:0]              reg_sw_e;         // row gain
    logic signed [5:0]              reg_sw_shift;     // 10 - e, registered so the apply cast starts from a flop (timing)
    logic                           latch_max;
    logic                           latch_log;

    // RMSNorm Pass 3 sign-guard (see RM_P3 below): true only when
    // ctrl_scalar came out positive during RM_P3/RM_P3_W/RM_P3_MUL_W.
    logic                           rm_offset_was_positive;

    assign req_mult_out     = reg_req_mult;
    assign req_shift_out    = reg_req_shift;
    assign scratch_a_out    = scratch_a;
    assign mm_num_row_tiles = reg_mm_rows;
    assign mm_num_ctiles    = reg_mm_ctiles;
    assign rms_log2n_out    = reg_rms_log2n;
    assign inv_rms_out      = reg_inv_rms;
    assign sw_shift_out     = reg_sw_shift;
    assign rms_growth_out   = rm_offset_was_positive;

    always_comb begin
        mmio_rdata = '0;
        if (mmio_addr == ADDR_STATUS)    mmio_rdata = {30'd0, reg_done, reg_busy};
        if (mmio_addr == ADDR_REQ_MULT)  mmio_rdata = reg_req_mult;
        if (mmio_addr == ADDR_REQ_SHIFT) mmio_rdata = {26'd0, reg_req_shift};
        if (mmio_addr == ADDR_SW_E)      mmio_rdata = {{27{reg_sw_e[4]}}, reg_sw_e};
        if (mmio_addr == ADDR_SM_Z)      mmio_rdata = {{24{sm_z[W-1]}}, sm_z};
    end


    // Softmax tile z = max + ln(S), saturated to W bits: exactly the offset SM_P4 subtracts (ctrl_scalar_sub_val),
    // exposed so the CPU can combine tiles (logsumexp over z). reg_scalar_max / reg_scalar_log hold their values
    // after the command, so this stays valid until the next softmax / RMSNorm command.
    logic signed [W:0]   sm_z_wide;
    always_comb begin
        sm_z_wide = $signed({reg_scalar_max[W-1], reg_scalar_max}) + $signed({reg_scalar_log[W-1], reg_scalar_log});
        if (sm_z_wide > SUB_VAL_MAX)      sm_z = SUB_VAL_MAX[W-1:0];
        else if (sm_z_wide < SUB_VAL_MIN) sm_z = SUB_VAL_MIN[W-1:0];
        else                              sm_z = sm_z_wide[W-1:0];
    end

    // ---- SwiGLU helpers ----
    // silu_s = sat12((wide + 2) >> 2): drop the 2 low bits of x*sigma (Q.10) with rounding.
    function automatic logic signed [SA_W-1:0] silu_round(input logic signed [WIDE_W-1:0] w);
        logic signed [WIDE_W:0] r;
        r = ($signed({w[WIDE_W-1], w}) + (WIDE_W+1)'(2)) >>> 2;   // + half of the 2 dropped bits
        if (r > ((1 <<< (SA_W-1)) - 1))      silu_round = (1 <<< (SA_W-1)) - 1;
        else if (r < -(1 <<< (SA_W-1)))      silu_round = -(1 <<< (SA_W-1));
        else                                 silu_round = SA_W'(r);
    endfunction

    // Row gain: largest e in [-8,10] with (4*max) * 2^e <= 127*2^10, i.e. floor(log2(127*2^10 / (4*max))). The cap at 10 keeps
    // the apply shift (10 - e) non-negative: no left shift in the cast (real rows have e in [-3, 2]).
    function automatic logic signed [4:0] sw_gain_fn(input logic [WIDE_W-1:0] maxp);
        logic [47:0] mx;
        logic [47:0] lim;
        logic signed [4:0] e;
        mx = {maxp, 2'b00};
        if (mx == 0) mx = 1;
        e = -5'sd8;
        for (int k = -8; k <= 10; k++) begin
            lim = 48'd130048;                            // 127 << 10
            if (k >= 0) begin if ((mx << k) <= lim)  e = 5'(k); end
            else        begin if (mx <= (lim << (-k))) e = 5'(k); end
        end
        sw_gain_fn = e;
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_sw_max <= '0;
            reg_sw_e   <= '0;
            reg_sw_shift <= 6'sd10;
        end else begin
            if (latch_sw_max && vpu_max_out[WIDE_W-1:0] > reg_sw_max) reg_sw_max <= vpu_max_out[WIDE_W-1:0];   // piped_max of |prod|
            if (latch_sw_gain) begin
                reg_sw_e   <= sw_gain_fn(reg_sw_max);
                reg_sw_shift <= 6'sd10 - 6'(sw_gain_fn(reg_sw_max));
                reg_sw_max <= '0;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (write_silu) for (int k = 0; k < N; k++) scratch_a[k] <= silu_round(vpu_wide_out[k]);
        if (write_scratch_a) for (int k = 0; k < N; k++) scratch_a[k] <= SA_W'(vpu_data_out[k]);
        if (latch_inv) reg_inv_rms <= vpu_inv_rms;
        if (latch_max)       reg_scalar_max <= vpu_max_out[W-1:0];
        if (latch_log)       reg_scalar_log <= vpu_data_out[0];
    end

    // Expanded FSM to handle chained pipelining (Backbone -> VecMul/MAC)
    typedef enum logic [5:0] {
        IDLE,
        // Streaming matmul (CMD=0x04)
        MM_START, MM_WAIT,
        // Softmax
        SM_P1, SM_P1_W,
        SM_P2, SM_P2_W1, SM_P2_W2,
        SM_P3, SM_P3_W,
        SM_P4, SM_P4_W, SM_P4_MUL_W,
        // RMSNorm
        RM_P1, RM_P1_W, RM_P1N, RM_P1N_W,
        RM_P2, RM_P2_W,
        RM_P3, RM_P3_W, RM_P3_MUL_W,
        RM_P4, RM_P4_W,
        // SwiGLU: cmd 5 silu group, 6 max group, 0xB row gain, 0xC apply group
        SW_P1, SW_P1_W,
        SW_P2, SW_P2_W, SW_P2_M,
        SW_G,
        SW_P4, SW_P4_W,
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
            reg_rms_log2n <= '0;
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
                if (mmio_addr == ADDR_RMS_LOG2N) reg_rms_log2n <= mmio_wdata[4:0];
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
        latch_max           = 1'b0;
        latch_log           = 1'b0;
        write_silu          = 1'b0;
        latch_sw_max        = 1'b0;
        mux_max_sel         = 1'b0;
        latch_sw_gain       = 1'b0;
        latch_inv           = 1'b0;
        
        src_sram_a_sel      = 1'b0;
        
        mux_bb_in_sel       = 3'b011;
        mux_mac_a_sel       = 2'b00;
        mux_mac_b_sel       = 2'b00;
        mux_vecmul_a_sel    = 2'b00;
        mux_vecmul_b_sel    = 3'b000;
        mux_vpu_out_sel     = 3'b000;
        
        en_piped_max        = 1'b0;
        en_mac_valid        = 1'b0;
        en_vecmul_valid     = 1'b0;
        en_bb_valid         = 1'b0;
        
        mac_clear_acc       = 1'b0;
        mac_op_mode         = 2'd0;
        vecmul_op_mode      = 2'd0;
        vecmul_scale_mode   = 3'b000;
        bb_shift_mode       = 1'b0;

        bb_bypass_ln        = 1'b0;
        bb_mode_post_ln     = 2'b00;
        bb_ln_pre_scaled    = 1'b0;
        ctrl_scalar_sub_val = '0;
        rm_offset_was_positive = 1'b0;

        case (state)
            IDLE: begin
                reg_busy = 1'b0;
                if (cmd_trigger == 8'h01) next_state = SM_P1;
                if (cmd_trigger == 8'h07) next_state = RM_P1;     // RMSNorm sum-of-squares, first tile
                if (cmd_trigger == 8'h08) next_state = RM_P1N;    // RMSNorm sum-of-squares, next tile
                if (cmd_trigger == 8'h09) next_state = RM_P2;     // RMSNorm scale (once per row)
                if (cmd_trigger == 8'h0A) next_state = RM_P4;     // RMSNorm apply (per tile)
                if (cmd_trigger == 8'h04) next_state = MM_START;
                if (cmd_trigger == 8'h05) next_state = SW_P1;     // SwiGLU: silu group
                if (cmd_trigger == 8'h06) next_state = SW_P2;     // SwiGLU: max-tracking group
                if (cmd_trigger == 8'h0B) next_state = SW_G;      // SwiGLU: row gain
                if (cmd_trigger == 8'h0C) next_state = SW_P4;     // SwiGLU: apply group
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
                bb_bypass_ln = 1'b1;
                ctrl_scalar_sub_val = sm_z;   // saturating max + ln(S), see sm_z above
                mux_bb_in_sel = 3'b010;
                vecmul_op_mode = 2'd2; vecmul_scale_mode = 3'b011; // UU, Q0.8 unsigned (saturating)
                mux_vecmul_a_sel = 2'b11; mux_vecmul_b_sel = 3'b010;
                mux_vpu_out_sel = 3'b000;

                if (state == SM_P4) begin en_bb_valid = 1; next_state = SM_P4_W; end
                else if (state == SM_P4_W && vpu_bb_valid) begin en_vecmul_valid = 1; next_state = SM_P4_MUL_W; end
                else if (state == SM_P4_MUL_W && vpu_vecmul_valid) begin write_ext_sram = 1; next_state = DONE; end
            end

            // =========================================================
            // RMSNORM SEQUENCE (Fixed 4-Pass Math Route)
            // =========================================================
            // One MAC beat of x*x (8 lanes); S accumulates across tiles. RM_P1 (cmd 7) clears first,
            // RM_P1N (cmd 8) accumulates onto the running sum.
            RM_P1, RM_P1_W, RM_P1N, RM_P1N_W: begin
                mux_mac_a_sel = 2'b00; mux_mac_b_sel = 2'b10; // Route X * X
                mac_op_mode = 2'd0; // SS
                if (state == RM_P1)       begin mac_clear_acc = 1; en_mac_valid = 1; next_state = RM_P1_W; end
                else if (state == RM_P1N) begin                    en_mac_valid = 1; next_state = RM_P1N_W; end
                else if (vpu_mac_valid) next_state = DONE;
            end

            // mk = round(S >> (FRAC_W+log2N-2)) -> shared ln at BITS=FRAC_W+2 -> rounded -0.5x -> reg_scalar_log
            RM_P2, RM_P2_W: begin
                bb_shift_mode = 1'b1;       // mean of squares (see tr_nonlinear_vpu rms_s_sat)
                bb_ln_pre_scaled = 1'b1;    // mk already has FRAC_W+2 fractional bits
                mux_bb_in_sel = 3'b000; bb_mode_post_ln = 2'b11; // rounded -0.5x
                mux_vpu_out_sel = 3'b001;
                if (state == RM_P2) begin en_bb_valid = 1; next_state = RM_P2_W; end
                else if (state == RM_P2_W && vpu_bb_valid) begin latch_log = 1; next_state = RM_P3; end
            end

            // exp(reg_scalar_log) through the (decay-only) backbone, sign-guarded; the raw e_a*mantissa
            // product is turned into the wide inv_rms register (reciprocal ROM on the growth branch).
            RM_P3, RM_P3_W, RM_P3_MUL_W: begin
                logic signed [W-1:0] rm_ctrl_scalar_raw;
                bb_bypass_ln = 1'b1;

                // Sign-guard: the shared exp backbone only supports non-positive inputs, but -0.5*ln(mean)
                // goes positive for small mean(x^2). Force it non-positive and remember the flip in
                // rm_offset_was_positive, which selects the reciprocal path at the latch_inv point.
                rm_ctrl_scalar_raw = reg_scalar_log;
                rm_offset_was_positive = ~rm_ctrl_scalar_raw[W-1] && (rm_ctrl_scalar_raw != '0);
                ctrl_scalar_sub_val = rm_offset_was_positive ? -rm_ctrl_scalar_raw : rm_ctrl_scalar_raw;

                src_sram_a_sel = 1; // Force 0 by reading empty Scratch A
                mux_bb_in_sel = 3'b100;
                vecmul_op_mode = 2'd2; vecmul_scale_mode = 3'b010; // UU; raw product is read from vpu_exp_raw
                mux_vecmul_a_sel = 2'b11; mux_vecmul_b_sel = 3'b010;
                mux_vpu_out_sel = 3'b000;

                if (state == RM_P3) begin en_bb_valid = 1; next_state = RM_P3_W; end
                else if (state == RM_P3_W && vpu_bb_valid) begin en_vecmul_valid = 1; next_state = RM_P3_MUL_W; end
                else if (state == RM_P3_MUL_W && vpu_vecmul_valid) begin latch_inv = 1; next_state = DONE; end
            end

            // Apply: out = sat8(round(x * inv_rms >> (FRAC_W+4))), x from the requantizer path.
            RM_P4, RM_P4_W: begin
                src_sram_a_sel = 0;
                vecmul_op_mode = 2'd1; // SU (Signed X * Unsigned wide InvRMS)
                vecmul_scale_mode = 3'b101; // rounded >>(FRAC_W+4), signed saturate
                mux_vecmul_a_sel = 2'b00; mux_vecmul_b_sel = 3'b011; // b = inv_rms register (broadcast)
                mux_vpu_out_sel = 3'b000;

                if (state == RM_P4) begin en_vecmul_valid = 1; next_state = RM_P4_W; end
                else if (state == RM_P4_W && vpu_vecmul_valid) begin write_ext_sram = 1; next_state = DONE; end
            end

            // =========================================================
            // SWIGLU (see header): cmd 5, 6, 0xB, 0xC. x / up groups arrive on the requantizer path (sram_data_a).
            // =========================================================
            // cmd 5: silu_s = sat12(round(x * sigma_rom[x] >> 2)) -> scratch_a (wide). x signed, sigma unsigned (SU).
            SW_P1, SW_P1_W: begin
                src_sram_a_sel    = 1'b0;
                vecmul_op_mode    = 2'd1;        // SU
                mux_vecmul_a_sel  = 2'b00;
                mux_vecmul_b_sel  = 3'b100;      // sigma ROM
                mux_vpu_out_sel   = 3'b000;
                if (state == SW_P1) begin en_vecmul_valid = 1'b1; next_state = SW_P1_W; end
                else if (vpu_vecmul_valid) begin write_silu = 1'b1; next_state = DONE; end
            end

            // cmd 6 (pass 1): |silu_s * up| max-tracking; up signed, silu_s signed (SS). The products go through the
            // shared piped_max (|prod| select) and the group max is folded into reg_sw_max.
            SW_P2, SW_P2_W, SW_P2_M: begin
                src_sram_a_sel    = 1'b0;
                vecmul_op_mode    = 2'd0;        // SS
                mux_vecmul_a_sel  = 2'b00;
                mux_vecmul_b_sel  = 3'b101;      // wide silu (scratch_a)
                mux_vpu_out_sel   = 3'b000;
                if (state == SW_P2) begin en_vecmul_valid = 1'b1; next_state = SW_P2_W; end
                else if (state == SW_P2_W && vpu_vecmul_valid) next_state = SW_P2_M;
                else if (state == SW_P2_M) begin
                    mux_max_sel  = 1'b1;
                    en_piped_max = 1'b1;
                    if (vpu_max_valid) begin latch_sw_max = 1'b1; next_state = DONE; end
                end
            end

            // cmd 0xB: row gain e from the running max (consumes and clears it).
            SW_G: begin
                latch_sw_gain = 1'b1;
                next_state    = DONE;
            end

            // cmd 0xC (pass 2): y = sat8(round_shift(4 * silu_s * up, 10 - e)) -> external result.
            SW_P4, SW_P4_W: begin
                src_sram_a_sel    = 1'b0;
                vecmul_op_mode    = 2'd0;        // SS
                vecmul_scale_mode = 3'b110;      // SwiGLU apply (shift = sw_shift_out)
                mux_vecmul_a_sel  = 2'b00;
                mux_vecmul_b_sel  = 3'b101;      // wide silu (scratch_a)
                mux_vpu_out_sel   = 3'b000;
                if (state == SW_P4) begin en_vecmul_valid = 1'b1; next_state = SW_P4_W; end
                else if (vpu_vecmul_valid) begin write_ext_sram = 1'b1; next_state = DONE; end
            end

            DONE: begin
                reg_done   = 1'b1;
                reg_busy   = 1'b0;
                next_state = IDLE;
            end
        endcase
    end
endmodule