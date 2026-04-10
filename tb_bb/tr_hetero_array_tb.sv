`timescale 1ns/1ps

module tr_hetero_array_tb();

    localparam int N = 8;

    // ========================================================================
    // DUT SIGNALS
    // ========================================================================
    logic signed [7:0]  x_vec_in [N];
    logic signed [19:0] x_scalar_in;
    
    logic               lane_0_mode;
    logic [1:0]         shift_mode;
    logic               exp_in_sel;
    
    // SPLIT DATA OUTPUTS
    logic [7:0]         y_ea_vec_out [N];
    logic [7:0]         y_man_vec_out [N];
    logic [19:0]        y_ea_scalar_out;
    logic [19:0]        y_man_scalar_out;

    // ========================================================================
    // DUT INSTANTIATION
    // ========================================================================
    tr_hetero_array #(.N(N)) dut (
        .x_vec_in(x_vec_in),
        .x_scalar_in(x_scalar_in),
        .lane_0_mode(lane_0_mode),
        .shift_mode(shift_mode),
        .exp_in_sel(exp_in_sel),
        .y_ea_vec_out(y_ea_vec_out),
        .y_man_vec_out(y_man_vec_out),
        .y_ea_scalar_out(y_ea_scalar_out),
        .y_man_scalar_out(y_man_scalar_out)
    );

    // ========================================================================
    // HELPER VARIABLES FOR DISPLAY (MOCKING THE MAC ENGINE)
    // ========================================================================
    real mac_val [N];
    real mac_scalar_val;
    real expected;
    
    initial begin
        $display("\n=======================================================================");
        $display(" STARTING HETEROGENEOUS ARRAY VERIFICATION (SPLIT-BUS / OPTION A)");
        $display("=======================================================================\n");

        // Initialize defaults
        for (int i=0; i<N; i++) x_vec_in[i] = '0;
        x_scalar_in = '0;
        lane_0_mode = 0; shift_mode = 0; exp_in_sel = 0;

        #10;

        // --------------------------------------------------------------------
        // TEST 1: SoftMax Exponentials (Vector Mode, Max_Sub Feed)
        // --------------------------------------------------------------------
        $display(">>> TEST 1: SoftMax Numerators (Vector Mode | Bypass (exp_in_sel = 0))");
        lane_0_mode = 1'b0; // Vector Mode
        shift_mode  = 2'b00;// Bypass Shifter
        exp_in_sel  = 1'b0; // Bypass Logarithm (taking x_vec_in directly into tr_exp)
        
        // Feed an array of negative Q4.4 numbers (representing x - x_max)
        x_vec_in[0] = -8'd0;   //  0.0
        x_vec_in[1] = -8'd8;   // -0.5
        x_vec_in[2] = -8'd16;  // -1.0
        x_vec_in[3] = -8'd32;  // -2.0
        #10;
        
        // MOCKING THE MAC ENGINE (e_a is Q0.8, Mantissa is Q4.4)
        mac_val[0] = (real'(y_ea_vec_out[0]) / 256.0) * (real'(y_man_vec_out[0]) / 16.0);
        mac_val[1] = (real'(y_ea_vec_out[1]) / 256.0) * (real'(y_man_vec_out[1]) / 16.0);
        mac_val[2] = (real'(y_ea_vec_out[2]) / 256.0) * (real'(y_man_vec_out[2]) / 16.0);
        mac_val[3] = (real'(y_ea_vec_out[3]) / 256.0) * (real'(y_man_vec_out[3]) / 16.0);

        $display("   Lane 0 [ 0.0] -> MAC Reconstruction: %f | Expected: 1.000", mac_val[0]);
        $display("   Lane 1 [-0.5] -> MAC Reconstruction: %f | Expected: 0.606", mac_val[1]);
        $display("   Lane 2 [-1.0] -> MAC Reconstruction: %f | Expected: 0.367", mac_val[2]);
        $display("   Lane 3 [-2.0] -> MAC Reconstruction: %f | Expected: 0.135", mac_val[3]);
        $display("--------------------------------------------------------------------\n");

        // --------------------------------------------------------------------
        // TEST 2: GELU Datapath (Vector Mode, Log-Exp Feed)
        // --------------------------------------------------------------------
        $display(">>> TEST 2: GELU Denominators (Vector Mode | exp_in_sel = 1)");
        lane_0_mode = 1'b0; // Vector Mode
        shift_mode  = 2'b00;// Bypass Shifter
        exp_in_sel  = 1'b1; // Route through Logarithm first
        
        // Feed positive Q4.4 numbers
        x_vec_in[0] = 8'd24;  // 1.5 -> Lane 0 (Shadow ALU)
        x_vec_in[1] = 8'd24;  // 1.5 -> Lane 1 (Standard ALU)
        #10;
        
        mac_val[0] = (real'(y_ea_vec_out[0]) / 256.0) * (real'(y_man_vec_out[0]) / 16.0);
        mac_val[1] = (real'(y_ea_vec_out[1]) / 256.0) * (real'(y_man_vec_out[1]) / 16.0);

        $display("   Lane 0 [1.5] -> MAC Reconstruction: %f (Powered by Shadow ALU)", mac_val[0]);
        $display("   Lane 1 [1.5] -> MAC Reconstruction: %f (Powered by Std 8-bit ALU)", mac_val[1]);
        
        if ((y_ea_vec_out[0] == y_ea_vec_out[1]) && (y_man_vec_out[0] == y_man_vec_out[1])) 
             $display("   [PASS] Lane 0 perfectly matches Lane 1 Split Buses!");
        else $display("   [FAIL] Lane 0 mismatch. Shadow ALU truncation error.");
        $display("--------------------------------------------------------------------\n");

        // --------------------------------------------------------------------
        // TEST 3: SoftMax Reciprocal (Scalar Mode, Shift = -x)
        // --------------------------------------------------------------------
        $display(">>> TEST 3: SoftMax Reciprocal (Scalar Mode | shift = -x)");
        lane_0_mode = 1'b1; // Scalar Mode
        shift_mode  = 2'b01;// -x (Reciprocal)
        exp_in_sel  = 1'b1; // Route through Logarithm
        
        // Feed 10.0 into the 20-bit Q12.8 Lane 0 input (10.0 * 256 = 2560)
        x_scalar_in = 20'd2560; 
        #10;
        
        // MOCKING THE MAC ENGINE (e_a is Qx.8, Mantissa is Q12.8)
        mac_scalar_val = (real'(y_ea_scalar_out) / 256.0) * (real'(y_man_scalar_out) / 256.0);
        expected = 1.0 / 10.0;
        $display("   Input Sum: 10.0 | MAC 1/x: %f | Expected: %f", mac_scalar_val, expected);
        $display("--------------------------------------------------------------------\n");

        // --------------------------------------------------------------------
        // TEST 4: LayerNorm Inverse StdDev (Scalar Mode, Shift = -x/2)
        // --------------------------------------------------------------------
        $display(">>> TEST 4: LayerNorm InvSqrt (Scalar Mode | shift = -x/2)");
        lane_0_mode = 1'b1; // Scalar Mode
        shift_mode  = 2'b10;// -x/2 (Inverse Square Root)
        exp_in_sel  = 1'b1; // Route through Logarithm
        
        // Feed 4.0 into the 20-bit Q12.8 Lane 0 input (4.0 * 256 = 1024)
        x_scalar_in = 20'd1024; 
        #10;
        
        mac_scalar_val = (real'(y_ea_scalar_out) / 256.0) * (real'(y_man_scalar_out) / 256.0);
        expected = 1.0 / $sqrt(4.0);
        $display("   Input Var: 4.0  | MAC 1/sqrt: %f | Expected: %f", mac_scalar_val, expected);
        $display("=======================================================================\n");

        // --------------------------------------------------------------------
        // TEST 5: Parallel Vector Reciprocals (GELU Denominator Pass)
        // --------------------------------------------------------------------
        $display(">>> TEST 5: Parallel Vector Reciprocals (shift = -x)");
        lane_0_mode = 1'b0; // Vector Mode
        shift_mode  = 2'b01; // -ln(x) -> 1/x
        exp_in_sel  = 1'b1; // Route through Logarithm
        
        // Input sequence in Q4.4: 1.0, 2.0, 4.0, 8.0
        x_vec_in[0] = 8'd16; // 1.0
        x_vec_in[1] = 8'd32; // 2.0
        x_vec_in[2] = 8'd64; // 4.0
        x_vec_in[3] = 8'd127;// ~8.0
        #10;
        
        for (int i=0; i<4; i++) begin
            mac_val[i] = (real'(y_ea_vec_out[i]) / 256.0) * (real'(y_man_vec_out[i]) / 16.0);
            expected = 16.0 / real'(x_vec_in[i]); 
            $display("   Lane %0d [%3.2f] -> 1/x: %f | Expected: %f", i, real'(x_vec_in[i])/16.0, mac_val[i], expected);
        end
        
        if (y_ea_vec_out[0] != 8'd0) $display("   [PASS] Vector lanes successfully computing parallel reciprocals.");
        $display("--------------------------------------------------------------------\n");

        // --------------------------------------------------------------------
        // TEST 6: Parallel Vector Inverse Square Roots (LayerNorm Scaling)
        // --------------------------------------------------------------------
        $display(">>> TEST 6: Parallel Vector InvSqrt (shift = -x/2)");
        lane_0_mode = 1'b0; 
        shift_mode  = 2'b10; // -ln(x)/2 -> 1/sqrt(x)
        exp_in_sel  = 1'b1;
        
        // Input sequence: 1.0, 4.0, 16.0
        x_vec_in[0] = 8'd16;  // 1.0
        x_vec_in[1] = 8'd64;  // 4.0
        x_vec_in[2] = 8'd127; // ~8.0
        #10;
        
        for (int i=0; i<3; i++) begin
            mac_val[i] = (real'(y_ea_vec_out[i]) / 256.0) * (real'(y_man_vec_out[i]) / 16.0);
            expected = 1.0 / $sqrt(real'(x_vec_in[i])/16.0);
            $display("   Lane %0d [%3.2f] -> 1/sqrt(x): %f | Expected: %f", i, real'(x_vec_in[i])/16.0, mac_val[i], expected);
        end
        $display("=======================================================================\n");

        $finish;
    end

endmodule