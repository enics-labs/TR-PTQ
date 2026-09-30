`timescale 1ns/1ps

module tr_ln_alu_tb();

    // Hardware Parameters
    localparam int WIDTH     = 16;
    localparam int BITS      = 4;
    localparam int OUT_WIDTH = 8;

    // Interconnects
    logic [WIDTH-1:0]            xq;
    logic signed [OUT_WIDTH-1:0] yq;

    // Instantiate Unit Under Test
    tr_ln_alu #(
        .WIDTH(WIDTH), .BITS(BITS), .OUT_WIDTH(OUT_WIDTH)
    ) u_ln (
        .xq(xq), .yq(yq)
    );

    int file_in, file_out, num_vecs, dummy;
    int stimulus_raw;

    initial begin
        file_in = $fopen("inputs.txt", "r");
        file_out = $fopen("hdl_out.txt", "w");
        
        if (!file_in || !file_out) begin
            $display("[ERROR] Could not open IO files! Ensure inputs.txt exists in workspace.");
            $finish;
        end

        // Read total vector count
        dummy = $fscanf(file_in, "%0d\n", num_vecs);

        for (int i = 0; i < num_vecs; i++) begin
            dummy = $fscanf(file_in, "%d\n", stimulus_raw);
            xq = WIDTH'(stimulus_raw);
            
            #10; // Wait for the shift-and-add combinational logic to settle
            
            // Format output without spaces to match C++
            $fwrite(file_out, "%0d %0d\n", xq, $signed(yq));
        end

        $fclose(file_in);
        $fclose(file_out);
        $display("[RTL SIM] Successfully evaluated %0d LN vectors.", num_vecs);
        $finish;
    end
endmodule