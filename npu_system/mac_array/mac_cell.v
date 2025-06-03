module mac_cell (
    input wire clk,
    input wire rst_n,
    input wire [15:0] a,      // BF16 input
    input wire [15:0] b,      // BF16 input
    input wire [15:0] c_in,   // Accumulation input
    output reg [15:0] c_out   // Accumulation output
);

    // Internal signals
    wire [15:0] mult_result;
    reg [15:0] acc_reg;
    
    // Instantiate BF16 multiplier
    bf16_multiplier mult (
        .clk(clk),
        .rst_n(rst_n),
        .a(a),
        .b(b),
        .result(mult_result)
    );
    
    // Accumulation logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= 16'd0;
            c_out <= 16'd0;
        end else begin
            // Add multiplication result to accumulation register
            acc_reg <= mult_result + c_in;
            c_out <= acc_reg;
        end
    end

endmodule 