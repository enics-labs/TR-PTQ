`timescale 1ns/1ps

module transformer_core_top import transformer_ctrl_pkg::*; #(
    parameter int N     = 8,
    parameter int W     = 8,
    parameter int ACC_W = 32,
    parameter int FRAC  = 4
)(
    input  logic             clk,
    input  logic             rst_n,

    // ==========================================
    // SOFTWARE / HOST INTERFACE
    // ==========================================
    input  logic    host_req_valid,
    input  opcode_e host_opcode,
    output logic    host_req_ready,
    output logic    host_done_pulse,

    // ==========================================
    // DATA INTERFACE (Memory / FIFOs)
    // ==========================================
    // Note: In a real FPGA, these would be BRAM read ports or AXI-Streams
    input  logic signed [W-1:0]     a_in [N],
    input  logic signed [W-1:0]     b_in [N],
    
    output logic                    out_valid,
    output logic signed [ACC_W-1:0] out_vec [N],
    output logic signed [ACC_W-1:0] out_dot
);

    // Internal Control Bus
    micro_op_t ctrl_bus;
    logic      dp_in_valid;
    logic      dp_mac_in_valid;

    // 1. Control Unit
    transformer_core_ctrl #(
        .MAX_LATENCY($clog2(N) - 1)
    ) u_ctrl (
        .clk             (clk),
        .rst_n           (rst_n),
        .host_req_valid  (host_req_valid),
        .host_opcode     (host_opcode),
        .host_req_ready  (host_req_ready),
        .host_done_pulse (host_done_pulse),
        .ctrl_bus        (ctrl_bus),
        .dp_in_valid     (dp_in_valid),
        .dp_mac_in_valid (dp_mac_in_valid)
    );

    // 2. The Datapath
    transformer_core_datapath #(
        .N(N), .W(W), .ACC_W(ACC_W), .FRAC(FRAC)
    ) u_datapath (
        .clk                 (clk), 
        .rst_n               (rst_n),
        .in_valid            (dp_in_valid), // Driven by FSM
        .a                   (a_in), 
        .b                   (b_in),
        .out_valid           (out_valid), 
        .out_vec             (out_vec), 
        .out_dot             (out_dot),
        
        // Unpack the Control Bus
        .ctrl_tr_lane0_mode  (ctrl_bus.tr_lane0_mode),
        .ctrl_tr_shift_mode  (ctrl_bus.tr_shift_mode),
        .ctrl_tr_exp_sel     (ctrl_bus.tr_exp_sel),
        .ctrl_mux_sub_val_sel(ctrl_bus.mux_sub_val_sel),
        .ctrl_mux_tr_vec_sel (ctrl_bus.mux_tr_vec_sel),
        .ctrl_mux_a_sel      (ctrl_bus.mux_a_sel),
        .ctrl_mux_b_sel      (ctrl_bus.mux_b_sel),
        .ctrl_mac_op_mode    (ctrl_bus.mac_op_mode),
        .ctrl_mac_elemwise   (ctrl_bus.mac_elemwise),
        .ctrl_mac_clear_acc  (ctrl_bus.mac_clear_acc),
        .ctrl_mac_in_valid   (dp_mac_in_valid), // Driven by FSM
        .ctrl_save_mean      (ctrl_bus.save_mean),
        .ctrl_save_sum       (ctrl_bus.save_sum),
        .ctrl_gelu_mode      (ctrl_bus.gelu_mode)
    );

endmodule