`timescale 1ns/1ps

module mac_array_tb;
    parameter ARRAY_SIZE = 32;
    parameter CLK_PERIOD = 10;  // 100MHz clock
    
    // Clock and reset
    reg clk;
    reg rst_n;
    
    // Input and output matrices
    reg [15:0] a_matrix [0:ARRAY_SIZE-1];
    reg [15:0] b_matrix [0:ARRAY_SIZE-1];
    wire [15:0] c_matrix [0:ARRAY_SIZE-1];
    
    // Instantiate MAC array
    mac_array #(
        .ARRAY_SIZE(ARRAY_SIZE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .a_matrix(a_matrix),
        .b_matrix(b_matrix),
        .c_matrix(c_matrix)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Test stimulus
    initial begin
        // Initialize
        rst_n = 0;
        for (integer i = 0; i < ARRAY_SIZE; i = i + 1) begin
            a_matrix[i] = 16'h3F80;  // 1.0 in BF16
            b_matrix[i] = 16'h3F80;  // 1.0 in BF16
        end
        
        // Release reset
        #(CLK_PERIOD*2);
        rst_n = 1;
        
        // Wait for computation
        #(CLK_PERIOD*100);
        
        // Check results
        for (integer i = 0; i < ARRAY_SIZE; i = i + 1) begin
            $display("Result[%0d] = %h", i, c_matrix[i]);
        end
        
        // End simulation
        #(CLK_PERIOD*10);
        $finish;
    end
    
    // Monitor
    initial begin
        $monitor("Time=%0t rst_n=%b", $time, rst_n);
    end

endmodule 