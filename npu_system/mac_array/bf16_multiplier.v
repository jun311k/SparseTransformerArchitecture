module bf16_multiplier (
    input wire clk,
    input wire rst_n,
    input wire [15:0] a,  // BF16 input
    input wire [15:0] b,  // BF16 input
    input wire in_valid,  // Input valid signal
    output reg out_valid, // Output valid signal
    output reg [15:0] result  // BF16 output
);

    // BF16 format: 1-bit sign, 8-bit exponent, 7-bit mantissa
    wire a_sign, b_sign;
    wire [7:0] a_exp, b_exp;
    wire [6:0] a_mant, b_mant;
    
    // Special case flags
    wire a_is_zero, b_is_zero;
    wire a_is_inf, b_is_inf;
    wire a_is_nan, b_is_nan;
    
    // Stage 1 registers
    reg stage1_sign;
    reg [8:0] stage1_exp;
    reg [13:0] stage1_mant;
    reg stage1_is_zero;
    reg stage1_is_inf;
    reg stage1_is_nan;
    reg stage1_valid;
    
    // Stage 2 registers
    reg stage2_sign;
    reg [7:0] stage2_exp;
    reg [13:0] stage2_mant_full;
    reg stage2_is_zero;
    reg stage2_is_inf;
    reg stage2_is_nan;
    reg stage2_valid;

    // For normalization (Verilog-2001 compatible)
    integer shift;
    
    // Extract components
    assign a_sign = a[15];
    assign b_sign = b[15];
    assign a_exp = a[14:7];
    assign b_exp = b[14:7];
    assign a_mant = a[6:0];
    assign b_mant = b[6:0];
    
    // Check special cases
    assign a_is_zero = (a_exp == 8'd0) && (a_mant == 7'd0);
    assign b_is_zero = (b_exp == 8'd0) && (b_mant == 7'd0);
    assign a_is_inf = (a_exp == 8'hFF) && (a_mant == 7'd0);
    assign b_is_inf = (b_exp == 8'hFF) && (b_mant == 7'd0);
    assign a_is_nan = (a_exp == 8'hFF) && (a_mant != 7'd0);
    assign b_is_nan = (b_exp == 8'hFF) && (b_mant != 7'd0);
    
    // Stage 1: Basic multiplication and exponent calculation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage1_sign <= 1'b0;
            stage1_exp <= 9'd0;
            stage1_mant <= 14'd0;
            stage1_is_zero <= 1'b0;
            stage1_is_inf <= 1'b0;
            stage1_is_nan <= 1'b0;
            stage1_valid <= 1'b0;
        end
        else begin
            stage1_valid <= in_valid;
            if (in_valid) begin
                // Sign calculation
                stage1_sign <= a[15] ^ b[15];
                
                // Special cases
                if (a_is_nan || b_is_nan) begin
                    stage1_is_nan <= 1'b1;
                    stage1_is_zero <= 1'b0;
                    stage1_is_inf <= 1'b0;
                    stage1_exp <= 9'd0;
                    stage1_mant <= 14'd0;
                end
                else if (a_is_inf || b_is_inf) begin
                    if (a_is_zero || b_is_zero) begin
                        stage1_is_nan <= 1'b1;
                        stage1_is_zero <= 1'b0;
                        stage1_is_inf <= 1'b0;
                    end else begin
                        stage1_is_inf <= 1'b1;
                        stage1_is_zero <= 1'b0;
                        stage1_is_nan <= 1'b0;
                    end
                    stage1_exp <= 9'd0;
                    stage1_mant <= 14'd0;
                end
                else if (a_is_zero || b_is_zero) begin
                    stage1_is_zero <= 1'b1;
                    stage1_is_inf <= 1'b0;
                    stage1_is_nan <= 1'b0;
                    stage1_exp <= 9'd0;
                    stage1_mant <= 14'd0;
                end
                else begin
                    stage1_is_zero <= 1'b0;
                    stage1_is_inf <= 1'b0;
                    stage1_is_nan <= 1'b0;
                    
                    // Multiply mantissas first
                    stage1_mant <= {1'b1, a_mant} * {1'b1, b_mant};
                    
                    // Calculate exponent
                    if (a_exp == 0 && b_exp == 0) begin
                        // Both inputs are denormalized
                        stage1_exp <= 9'd0;
                    end
                    else if (a_exp == 0) begin
                        stage1_exp <= {1'b0, b_exp} - 9'd127;
                    end
                    else if (b_exp == 0) begin
                        stage1_exp <= {1'b0, a_exp} - 9'd127;
                    end
                    else begin
                        stage1_exp <= {1'b0, a_exp} + {1'b0, b_exp} - 9'd127;
                    end
                end
            end
        end
    end

    // Stage 2: Normalization and overflow/underflow check
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stage2_sign <= 1'b0;
            stage2_exp <= 8'd0;
            stage2_mant_full <= 14'd0;
            stage2_is_zero <= 1'b0;
            stage2_is_inf <= 1'b0;
            stage2_is_nan <= 1'b0;
            stage2_valid <= 1'b0;
        end
        else begin
            stage2_valid <= stage1_valid;
            stage2_sign <= stage1_sign;
            stage2_is_nan <= stage1_is_nan;
            if (stage1_is_nan) begin
                stage2_exp <= 8'hFF;
                stage2_mant_full <= 14'h2000;
                stage2_is_zero <= 1'b0;
                stage2_is_inf <= 1'b0;
            end
            else if (stage1_is_inf) begin
                stage2_exp <= 8'hFF;
                stage2_mant_full <= 14'd0;
                stage2_is_zero <= 1'b0;
                stage2_is_inf <= 1'b1;
            end
            else if (stage1_is_zero || stage1_mant == 0) begin
                stage2_exp <= 8'd0;
                stage2_mant_full <= 14'd0;
                stage2_is_zero <= 1'b1;
                stage2_is_inf <= 1'b0;
            end
            else begin
                // Normalize mantissa
                reg [13:0] norm_mant;
                reg [8:0] norm_exp;
                norm_mant = stage1_mant;
                norm_exp = stage1_exp;
                
                // Normalize based on the most significant bit
                if (norm_mant[13]) begin
                    norm_mant = norm_mant >> 1;
                    norm_exp = norm_exp + 1;
                end
                
                // Overflow check (BF16 max exponent is 127)
                if (norm_exp >= 9'd128) begin
                    stage2_exp <= 8'hFF;
                    stage2_mant_full <= 14'd0;
                    stage2_is_inf <= 1'b1;
                    stage2_is_zero <= 1'b0;
                end
                // Underflow
                else if (norm_exp == 0) begin
                    stage2_exp <= 8'd0;
                    stage2_mant_full <= 14'd0;
                    stage2_is_zero <= 1'b1;
                    stage2_is_inf <= 1'b0;
                end else begin
                    stage2_exp <= norm_exp[7:0];
                    stage2_mant_full <= norm_mant;
                    stage2_is_zero <= 1'b0;
                    stage2_is_inf <= 1'b0;
                end
            end
        end
    end

    // Output assignment
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 16'd0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= stage2_valid;
            if (stage2_is_nan) begin
                result <= {1'b0, 8'hFF, 7'h40};  // Quiet NaN
            end
            else if (stage2_is_inf) begin
                result <= {stage2_sign, 8'hFF, 7'h00};  // Infinity
            end
            else if (stage2_is_zero) begin
                result <= {stage2_sign, 8'h00, 7'h00};  // Zero
            end
            else begin
                result <= {stage2_sign, stage2_exp, stage2_mant_full[12:6]};
            end
        end
    end

endmodule 