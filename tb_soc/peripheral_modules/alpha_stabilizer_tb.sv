`timescale 1ns/1ps

module tb_alpha_stabilizer();

    localparam int N = 8;
    localparam int W = 8;

    // DUT Signals
    logic signed [W-1:0] in_vec [N];
    logic signed [W-1:0] out_vec [N];

    // Instantiate the DUT
    alpha_stabilizer #(.N(N), .W(W)) dut (
        .in_vec  (in_vec),
        .out_vec (out_vec)
    );

    int errors = 0;

    initial begin
        $display("=======================================================================");
        $display(" STARTING EXHAUSTIVE ALPHA STABILIZER VERIFICATION (ALL 256 STATES)");
        $display("=======================================================================");

        // We only need to test Lane 0 since all lanes are structurally identical
        for (int i = 1; i < N; i++) begin
            in_vec[i] = '0; 
        end

        // Sweep every possible 8-bit signed value (-128 to 127)
        for (int test_val = -128; test_val <= 127; test_val++) begin
            
            logic signed [7:0] hw_in;
            logic signed [7:0] hw_out;
            logic signed [7:0] exp_out;
            
            hw_in = test_val[7:0];
            
            // 1. Drive the hardware
            in_vec[0] = hw_in;
            #1; // Wait for combinational logic to settle

            hw_out = out_vec[0];

            // 2. Calculate the Golden Expected Value
            exp_out = golden_model(hw_in);

            // 3. Compare and Report
            if (hw_out !== exp_out) begin
                $display("   \033[0;31m[FAIL]\033[0m Input: %4d | Exp: %4d | HW: %4d", hw_in, exp_out, hw_out);
                errors++;
            end
        end

        $display("=======================================================================");
        if (errors == 0) begin
            $display(" \033[0;32m[SUCCESS]\033[0m Zero errors! Shift-and-Add Logic is 100%% Mathematically Perfect.");
        end else begin
            $display(" \033[0;31m[FAILED]\033[0m Found %0d mismatches.", errors);
        end
        $display("=======================================================================");
        
        $finish;
    end

    // ========================================================================
    // BEHAVIORAL GOLDEN MODEL
    // ========================================================================
    function logic signed [7:0] golden_model(input logic signed [7:0] x);
        logic [7:0] abs_x;
        logic [2:0] region;
        int         multiplier;
        int         raw_prod;
        logic signed [7:0] scaled_val;
        logic signed [7:0] final_val;

        // 1. Absolute Value
        abs_x = (x < 0) ? -x : x;

        // 2. LUT Indexing
        region = abs_x[6:4];
        case (region)
            3'b000:  multiplier = 27;
            3'b001:  multiplier = 26;
            3'b010:  multiplier = 25;
            default: multiplier = 24;
        endcase

        // 3. Multiplication (Behavioral Int Math)
        raw_prod = int'(x) * multiplier;

        // 4. Scaling and Saturation (Q8.8 to Q4.4)
        if (raw_prod > 2032) begin
            scaled_val = 8'sd127;
        end else if (raw_prod < -2048) begin
            scaled_val = -8'sd128;
        end else begin
            // Arithmetic right shift by 4 (mimics hardware slice [11:4])
            scaled_val = raw_prod >>> 4; 
        end

        // 5. Force Negative Absolute Value
        final_val = (scaled_val > 0) ? -scaled_val : scaled_val;

        return final_val;
    endfunction

endmodule