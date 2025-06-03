module mac_array #(
    parameter ARRAY_SIZE = 32
)(
    input wire clk,
    input wire rst_n,
    input wire [15:0] a_matrix [0:ARRAY_SIZE-1],  // Input matrix A
    input wire [15:0] b_matrix [0:ARRAY_SIZE-1],  // Input matrix B
    output reg [15:0] c_matrix [0:ARRAY_SIZE-1]   // Output matrix C
);

    // Internal signals for MAC cell connections
    wire [15:0] mac_outputs [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    wire [15:0] mac_inputs [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    
    // Generate MAC cells
    genvar i, j;
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : row_gen
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : col_gen
                // First column gets input from a_matrix
                assign mac_inputs[i][j] = (j == 0) ? 16'd0 : mac_outputs[i][j-1];
                
                // Instantiate MAC cell
                mac_cell mac_inst (
                    .clk(clk),
                    .rst_n(rst_n),
                    .a(a_matrix[i]),
                    .b(b_matrix[j]),
                    .c_in(mac_inputs[i][j]),
                    .c_out(mac_outputs[i][j])
                );
            end
        end
    endgenerate
    
    // Output assignment
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < ARRAY_SIZE; i = i + 1) begin
                c_matrix[i] <= 16'd0;
            end
        end else begin
            for (integer i = 0; i < ARRAY_SIZE; i = i + 1) begin
                c_matrix[i] <= mac_outputs[i][ARRAY_SIZE-1];
            end
        end
    end

endmodule 