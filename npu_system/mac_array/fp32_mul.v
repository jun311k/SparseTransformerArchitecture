module fp32_mul (
    input wire          clk         ,
    input wire          rst_n       ,
    input wire [31:0]   a           ,       // FP32 input
    input wire [31:0]   b           ,       // FP32 input
    input wire          in_valid    ,       // Input valid signal
    output reg          out_valid   ,       // Output valid signal
    output reg [31:0]   result              // FP32 output
);

    localparam TOT_BITS = 32;
    localparam EXP_BITS = 8;
    localparam MAN_BITS = 23;

    localparam MAN_STT = 0;
    localparam MAN_END = MAN_BITS-1;
    localparam EXP_STT = MAN_BITS;
    localparam EXP_END = TOT_BITS - 2;
    localparam SGN_BIT = TOT_BITS -1;

    localparam FP_BIAS = 127;

    localparam [EXP_BITS-1:0] EXP_ZERO    = {EXP_BITS{1'b0}}; // 8'h00
    localparam [EXP_BITS-1:0] EXP_INF_NAN = {EXP_BITS{1'b1}}; // 8'hFF
    localparam [MAN_BITS-1:0] MAN_ZERO    = {MAN_BITS{1'b0}};

    // FP32 format: 1-bit sign, 8-bit exponent, 23-bit mantissa
    wire a_sign, b_sign;
    wire [EXP_BITS-1:0] a_exp, b_exp;
    wire [MAN_BITS-1:0] a_mant, b_mant;
    
    // Special case flags
    wire a_is_zero, b_is_zero;
    wire a_is_inf, b_is_inf;
    wire a_is_nan, b_is_nan;    

    // Extract components
    assign a_sign = a[SGN_BIT];
    assign b_sign = b[SGN_BIT];
    assign a_exp  = a[EXP_END:EXP_STT];
    assign b_exp  = b[EXP_END:EXP_STT];
    assign a_mant = a[MAN_END:MAN_STT];
    assign b_mant = b[MAN_END:MAN_STT];

    // Special case checks
    assign a_is_zero = (a_exp == 8'h00) && (a_mant == 23'h000000);  // exponent is 0 and mantissa is 0
    assign b_is_zero = (b_exp == 8'h00) && (b_mant == 23'h000000);  // exponent is 0 and mantissa is 0
    assign a_is_inf  = (a_exp == 8'hff && a_mant == 23'h0);         // exponent is 255 and mantissa is 0
    assign b_is_inf  = (b_exp == 8'hff && b_mant == 23'h0);         // exponent is 255 and mantissa is 0
    assign a_is_nan  = (a_exp == 8'hff && a_mant != 23'h0);         // exponent is 255 and mantissa is not 0
    assign b_is_nan  = (b_exp == 8'hff && b_mant != 23'h0);         // exponent is 255 and mantissa is not 0
    
    // subnormal check
    wire a_is_subnormal, b_is_subnormal;
    assign a_is_subnormal = (a_exp == 8'h00 && a_mant != 23'h0); // zero signal is there. So this is only for non-zero case
    assign b_is_subnormal = (b_exp == 8'h00 && b_mant != 23'h0); // zero signal is there. So this is only for non-zero case

    
    // Significand (mantissa with implicit leading bit)
    // Leading bit is '0' for zero and subnormal numbers (exponent field is 0)
    // Leading bit is '1' for normalized numbers, Infinity, and NaN (exponent field is not 0)
    wire [23:0] a_mant_std, b_mant_std;
    wire a_std_leading_bit, b_std_leading_bit;

    assign a_std_leading_bit = (a_exp == 8'h00) ? 1'b0 : 1'b1;
    assign b_std_leading_bit = (b_exp == 8'h00) ? 1'b0 : 1'b1;
    assign a_mant_std = {a_std_leading_bit, a_mant};
    assign b_mant_std = {b_std_leading_bit, b_mant};



    // Signals for each stages
    
    // Pipeline Stage 1 -> Stage 2 Registers
    reg s1_in_valid_reg;
    reg s1_a_sign_reg, s1_b_sign_reg;
    reg [EXP_BITS-1:0] s1_a_exp_reg, s1_b_exp_reg;
    reg [MAN_BITS:0] s1_a_mant_std_reg, s1_b_mant_std_reg; // 24-bit (MAN_BITS+1)
    reg s1_a_is_zero_reg, s1_b_is_zero_reg;
    reg s1_a_is_inf_reg,  s1_b_is_inf_reg;
    reg s1_a_is_nan_reg,  s1_b_is_nan_reg;
    reg s1_a_is_subnormal_reg, s1_b_is_subnormal_reg;

    // Pipeline Stage 2 -> Stage 3 Registers
    reg s2_in_valid_reg;
    reg s2_res_sign_reg;
    reg signed [EXP_BITS+3:0] s2_res_exp_provisional_reg;  // Extended to 11 bits
    reg [2*MAN_BITS+1:0] s2_prod_mant_reg;
    reg s2_is_special_case_reg;
    reg [TOT_BITS-1:0] s2_special_result_word_reg;

    // STAGE 1: Decode and Prepare (Combinational part already done by assigns above)
    // Registering stage 1 outputs
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_in_valid_reg <= 1'b0;
            s1_a_sign_reg <= 1'b0; s1_b_sign_reg <= 1'b0;
            s1_a_exp_reg <= {EXP_BITS{1'b0}}; s1_b_exp_reg <= {EXP_BITS{1'b0}};
            s1_a_mant_std_reg <= {(MAN_BITS+1){1'b0}}; s1_b_mant_std_reg <= {(MAN_BITS+1){1'b0}};
            s1_a_is_zero_reg <= 1'b0; s1_b_is_zero_reg <= 1'b0;
            s1_a_is_inf_reg <= 1'b0;  s1_b_is_inf_reg <= 1'b0;
            s1_a_is_nan_reg <= 1'b0;  s1_b_is_nan_reg <= 1'b0;
            s1_a_is_subnormal_reg <= 1'b0; s1_b_is_subnormal_reg <= 1'b0;
        end else begin
            s1_in_valid_reg <= in_valid;
            if (in_valid) begin
                s1_a_sign_reg <= a_sign; s1_b_sign_reg <= b_sign;
                s1_a_exp_reg <= a_exp; s1_b_exp_reg <= b_exp;
                s1_a_mant_std_reg <= a_mant_std; s1_b_mant_std_reg <= b_mant_std;
                s1_a_is_zero_reg <= a_is_zero; s1_b_is_zero_reg <= b_is_zero;
                s1_a_is_inf_reg <= a_is_inf;  s1_b_is_inf_reg <= b_is_inf;
                s1_a_is_nan_reg <= a_is_nan;  s1_b_is_nan_reg <= b_is_nan;
                s1_a_is_subnormal_reg <= a_is_subnormal; s1_b_is_subnormal_reg <= b_is_subnormal;
            end
        end
    end

    // STAGE 2: Calculate
    wire s2_res_sign_comb;
    wire signed [EXP_BITS+3:0] s2_res_exp_provisional_comb;  // Extended to 11 bits
    wire [2*MAN_BITS+1:0] s2_prod_mant_comb;
    wire s2_is_special_case_comb;
    wire [TOT_BITS-1:0] s2_special_result_word_comb;

    wire [EXP_BITS-1:0] eff_a_exp_s1, eff_b_exp_s1;

    // Effective exponent for sum: if subnormal, use 1, else use stored exponent.
    // If zero, stored exponent is 0, subnormal is false, so eff_exp is 0.
    assign eff_a_exp_s1 = s1_a_is_subnormal_reg ? 8'd1 : s1_a_exp_reg;
    assign eff_b_exp_s1 = s1_b_is_subnormal_reg ? 8'd1 : s1_b_exp_reg;

    assign s2_res_sign_comb = s1_a_sign_reg ^ s1_b_sign_reg;
    // E_res_stored = E_a_stored_eff + E_b_stored_eff - bias
    assign s2_res_exp_provisional_comb = $signed({3'b000, eff_a_exp_s1}) + $signed({3'b000, eff_b_exp_s1}) - $signed({3'b000, FP_BIAS});
    assign s2_prod_mant_comb = s1_a_mant_std_reg * s1_b_mant_std_reg; // 24b * 24b = 48b

    // Special case determination for Stage 2
    wire s2_res_is_nan, s2_res_is_inf, s2_res_is_zero;
    wire s2_res_is_overflow;
    wire s2_res_is_underflow;
    
    // Check for overflow in exponent calculation
    assign s2_res_is_overflow = (s2_res_exp_provisional_comb > $signed({3'b000, EXP_INF_NAN}));
    
    // Check for underflow - include smallest normalized numbers multiplication
    assign s2_res_is_underflow = (s2_res_exp_provisional_comb <= -126) || 
                                ((s2_res_exp_provisional_comb == -125) && 
                                 (s1_a_exp_reg == 8'd1) && (s1_b_exp_reg == 8'd1));

    // Modified zero detection to handle subnormal numbers correctly
    assign s2_res_is_zero = (s1_a_is_zero_reg || s1_b_is_zero_reg || 
                            (s1_a_is_subnormal_reg && s1_b_is_subnormal_reg)) && 
                            !s2_res_is_nan && !s2_res_is_inf;

    assign s2_res_is_nan = s1_a_is_nan_reg || s1_b_is_nan_reg ||
                           (s1_a_is_inf_reg && s1_b_is_zero_reg) ||
                           (s1_b_is_inf_reg && s1_a_is_zero_reg);
    assign s2_res_is_inf = ((s1_a_is_inf_reg || s1_b_is_inf_reg) && !s2_res_is_nan) || s2_res_is_overflow;

    assign s2_is_special_case_comb = s2_res_is_nan || s2_res_is_inf || s2_res_is_zero || s2_res_is_underflow;

    // Default Quiet NaN: sign=0, exp=all 1s, mantissa MSB=1
    localparam [MAN_END:MAN_STT] QNAN_MANTISSA = (1'b1 << (MAN_BITS-1));
    assign s2_special_result_word_comb =
        s2_res_is_nan  ? {1'b0, EXP_INF_NAN, QNAN_MANTISSA} : // Default QNaN
        s2_res_is_inf  ? {s2_res_sign_comb, EXP_INF_NAN, MAN_ZERO} :
        (s2_res_is_zero || s2_res_is_underflow) ? {s2_res_sign_comb, EXP_ZERO, MAN_ZERO} :
                         32'b0; // Should not happen if s2_is_special_case_comb is true

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_in_valid_reg <= 1'b0;
            s2_res_sign_reg <= 1'b0;
            s2_res_exp_provisional_reg <= {(EXP_BITS+4){1'b0}};
            s2_prod_mant_reg <= {(2*MAN_BITS+2){1'b0}};
            s2_is_special_case_reg <= 1'b0;
            s2_special_result_word_reg <= {TOT_BITS{1'b0}};
        end else begin
            s2_in_valid_reg <= s1_in_valid_reg;
            if (s1_in_valid_reg) begin
                s2_res_sign_reg <= s2_res_sign_comb;
                s2_res_exp_provisional_reg <= s2_res_exp_provisional_comb;
                s2_prod_mant_reg <= s2_prod_mant_comb;
                s2_is_special_case_reg <= s2_is_special_case_comb;
                s2_special_result_word_reg <= s2_special_result_word_comb;
            end
        end
    end

    // STAGE 3: Normalize, Round, Assemble
    // Combinational logic for Stage 3
    wire [TOT_BITS-1:0] result_comb;
    
    // Normalization and Rounding logic (simplified: flush to zero on underflow)
    // Intermediate signals for normalization
    reg signed [EXP_BITS+3:0] norm_exp; // Extended to 11 bits for better overflow handling
    reg [2*MAN_BITS+1:0]      norm_mant_shifted; // 48 bits
    reg [MAN_BITS-1:0]        final_mant;
    reg [EXP_BITS-1:0]        final_exp;
    reg                       norm_prod_is_zero;
    reg signed [EXP_BITS+3:0] exp_after_norm_shift; // Extended to 11 bits
    reg [2*MAN_BITS+1:0]      mant_after_norm_shift;
    reg [5:0]                 left_shift_amount; // Max shift for 48-bit mantissa to bring MSB to bit 46

    always @(*) begin
        // Initialize outputs of the always block to avoid latches
        {exp_after_norm_shift, mant_after_norm_shift, left_shift_amount} = '0;
        // Step 1: Initial exponent and mantissa from Stage 2 product
        norm_exp = s2_res_exp_provisional_reg;
        norm_mant_shifted = s2_prod_mant_reg;
        norm_prod_is_zero = (s2_prod_mant_reg == 0); // If product of significands is zero

        // Step 2: Normalize mantissa (check for product >= 2.0 or < 1.0)
        // Right shift if product mantissa MSB (bit 47) is 1 (value >= 2.0 * 2^exp)
        if (norm_mant_shifted[2*MAN_BITS+1]) begin // Check bit 47
            exp_after_norm_shift = norm_exp + 1;
            mant_after_norm_shift = norm_mant_shifted >> 1;
        end else begin
            exp_after_norm_shift = norm_exp;
            mant_after_norm_shift = norm_mant_shifted;
        end

        // Left shift if mantissa is < 1.0 (mant_after_norm_shift[46] is 0)
        // This uses a generic priority encoder logic (can be synthesized from a loop)
        // left_shift_amount is assigned within the if/else if chain.
        if (!norm_prod_is_zero) begin // Only normalize if product is not zero
            if (mant_after_norm_shift[2*MAN_BITS]) begin // If bit 46 is 1, it's already normalized (1.xxxx)
                left_shift_amount = 6'd0;
            end else if (mant_after_norm_shift[2*MAN_BITS-1]) begin // If bit 45 is 1
                left_shift_amount = 6'd1;
            end else if (mant_after_norm_shift[2*MAN_BITS-2]) begin // If bit 44 is 1
                left_shift_amount = 6'd2;
            end else if (mant_after_norm_shift[2*MAN_BITS-3]) begin // If bit 43 is 1
                left_shift_amount = 6'd3;
            end else if (mant_after_norm_shift[2*MAN_BITS-4]) begin // If bit 42 is 1
                left_shift_amount = 6'd4;
            end else if (mant_after_norm_shift[2*MAN_BITS-5]) begin // If bit 41 is 1
                left_shift_amount = 6'd5;
            end else if (mant_after_norm_shift[2*MAN_BITS-6]) begin // If bit 40 is 1
                left_shift_amount = 6'd6;
            end else if (mant_after_norm_shift[2*MAN_BITS-7]) begin // If bit 39 is 1
                left_shift_amount = 6'd7;
            end else if (mant_after_norm_shift[2*MAN_BITS-8]) begin // If bit 38 is 1
                left_shift_amount = 6'd8;
            end else if (mant_after_norm_shift[2*MAN_BITS-9]) begin // If bit 37 is 1
                left_shift_amount = 6'd9;
            end else if (mant_after_norm_shift[2*MAN_BITS-10]) begin // If bit 36 is 1
                left_shift_amount = 6'd10;
            end else if (mant_after_norm_shift[2*MAN_BITS-11]) begin // If bit 35 is 1
                left_shift_amount = 6'd11;
            end else if (mant_after_norm_shift[2*MAN_BITS-12]) begin // If bit 34 is 1
                left_shift_amount = 6'd12;
            end else if (mant_after_norm_shift[2*MAN_BITS-13]) begin // If bit 33 is 1
                left_shift_amount = 6'd13;
            end else if (mant_after_norm_shift[2*MAN_BITS-14]) begin // If bit 32 is 1
                left_shift_amount = 6'd14;
            end else if (mant_after_norm_shift[2*MAN_BITS-15]) begin // If bit 31 is 1
                left_shift_amount = 6'd15;
            end else if (mant_after_norm_shift[2*MAN_BITS-16]) begin // If bit 30 is 1
                left_shift_amount = 6'd16;
            end else if (mant_after_norm_shift[2*MAN_BITS-17]) begin // If bit 29 is 1
                left_shift_amount = 6'd17;
            end else if (mant_after_norm_shift[2*MAN_BITS-18]) begin // If bit 28 is 1
                left_shift_amount = 6'd18;
            end else if (mant_after_norm_shift[2*MAN_BITS-19]) begin // If bit 27 is 1
                left_shift_amount = 6'd19;
            end else if (mant_after_norm_shift[2*MAN_BITS-20]) begin // If bit 26 is 1
                left_shift_amount = 6'd20;
            end else if (mant_after_norm_shift[2*MAN_BITS-21]) begin // If bit 25 is 1
                left_shift_amount = 6'd21;
            end else if (mant_after_norm_shift[2*MAN_BITS-22]) begin // If bit 24 is 1
                left_shift_amount = 6'd22;
            end else if (mant_after_norm_shift[2*MAN_BITS-23]) begin // If bit 23 is 1
                left_shift_amount = 6'd23;
            end else if (mant_after_norm_shift[2*MAN_BITS-24]) begin // If bit 22 is 1
                left_shift_amount = 6'd24;
            end else if (mant_after_norm_shift[2*MAN_BITS-25]) begin // If bit 21 is 1
                left_shift_amount = 6'd25;
            end else if (mant_after_norm_shift[2*MAN_BITS-26]) begin // If bit 20 is 1
                left_shift_amount = 6'd26;
            end else if (mant_after_norm_shift[2*MAN_BITS-27]) begin // If bit 19 is 1
                left_shift_amount = 6'd27;
            end else if (mant_after_norm_shift[2*MAN_BITS-28]) begin // If bit 18 is 1
                left_shift_amount = 6'd28;
            end else if (mant_after_norm_shift[2*MAN_BITS-29]) begin // If bit 17 is 1
                left_shift_amount = 6'd29;
            end else if (mant_after_norm_shift[2*MAN_BITS-30]) begin // If bit 16 is 1
                left_shift_amount = 6'd30;
            end else if (mant_after_norm_shift[2*MAN_BITS-31]) begin // If bit 15 is 1
                left_shift_amount = 6'd31;
            end else if (mant_after_norm_shift[2*MAN_BITS-32]) begin // If bit 14 is 1
                left_shift_amount = 6'd32;
            end else if (mant_after_norm_shift[2*MAN_BITS-33]) begin // If bit 13 is 1
                left_shift_amount = 6'd33;
            end else if (mant_after_norm_shift[2*MAN_BITS-34]) begin // If bit 12 is 1
                left_shift_amount = 6'd34;
            end else if (mant_after_norm_shift[2*MAN_BITS-35]) begin // If bit 11 is 1
                left_shift_amount = 6'd35;
            end else if (mant_after_norm_shift[2*MAN_BITS-36]) begin // If bit 10 is 1
                left_shift_amount = 6'd36;
            end else if (mant_after_norm_shift[2*MAN_BITS-37]) begin // If bit 9 is 1
                left_shift_amount = 6'd37;
            end else if (mant_after_norm_shift[2*MAN_BITS-38]) begin // If bit 8 is 1
                left_shift_amount = 6'd38;
            end else if (mant_after_norm_shift[2*MAN_BITS-39]) begin // If bit 7 is 1
                left_shift_amount = 6'd39;
            end else if (mant_after_norm_shift[2*MAN_BITS-40]) begin // If bit 6 is 1
                left_shift_amount = 6'd40;
            end else if (mant_after_norm_shift[2*MAN_BITS-41]) begin // If bit 5 is 1
                left_shift_amount = 6'd41;
            end else if (mant_after_norm_shift[2*MAN_BITS-42]) begin // If bit 4 is 1
                left_shift_amount = 6'd42;
            end else if (mant_after_norm_shift[2*MAN_BITS-43]) begin // If bit 3 is 1
                left_shift_amount = 6'd43;
            end else if (mant_after_norm_shift[2*MAN_BITS-44]) begin // If bit 2 is 1
                left_shift_amount = 6'd44;
            end else if (mant_after_norm_shift[2*MAN_BITS-45]) begin // If bit 1 is 1
                left_shift_amount = 6'd45;
            end else if (mant_after_norm_shift[2*MAN_BITS-46]) begin // If bit 0 is 1
                left_shift_amount = 6'd46;
            end else begin // All zeros
                left_shift_amount = (2*MAN_BITS)+1; // Effectively zero
            end
        end

        // Apply left shift based on calculated amount
        if (left_shift_amount > 0 && left_shift_amount <= (2*MAN_BITS)+1) begin
            exp_after_norm_shift = exp_after_norm_shift - left_shift_amount;
            mant_after_norm_shift = mant_after_norm_shift << left_shift_amount;
        end
    end

    // Step 3: Rounding (Round to Nearest, Ties to Even)
    // After normalization, mant_after_norm_shift[46] is the implicit '1'.
    // Fraction is mant_after_norm_shift[45:0]. We need MAN_BITS (23) of these.
    wire [MAN_BITS-1:0] unrounded_fraction = mant_after_norm_shift[(2*MAN_BITS)-1 : (2*MAN_BITS)-MAN_BITS]; // bits [45:23]
    wire guard_bit  = mant_after_norm_shift[(2*MAN_BITS)-MAN_BITS-1]; // bit 22
    wire round_bit  = mant_after_norm_shift[(2*MAN_BITS)-MAN_BITS-2]; // bit 21
    wire sticky_bit = |(mant_after_norm_shift[(2*MAN_BITS)-MAN_BITS-3 : 0]); // OR of bits [20:0]

    wire lsb_of_unrounded = unrounded_fraction[0];
    wire round_increment = guard_bit && ( (round_bit || sticky_bit) || (!(round_bit || sticky_bit) && lsb_of_unrounded) );

    wire [MAN_BITS:0] rounded_fraction_sum = {1'b0, unrounded_fraction} + round_increment;
    
    // Step 4: Final exponent check (Overflow/Underflow) and final mantissa assignment
    always @(*) begin
        // Default assignments
        final_exp = exp_after_norm_shift[EXP_BITS-1:0]; // Default to normalized exponent
        final_mant = rounded_fraction_sum[MAN_BITS-1:0]; // Default to rounded mantissa

        // First check for underflow
        if (exp_after_norm_shift <= -126) begin // Underflow (Flush to Zero for simplicity)
            final_exp = EXP_ZERO;
            final_mant = MAN_ZERO;
        end
        // Then check for zero cases
        else if (norm_prod_is_zero || (left_shift_amount > (2*MAN_BITS)) || 
                (s1_a_is_subnormal_reg && s1_b_is_subnormal_reg)) begin
            final_exp = EXP_ZERO;
            final_mant = MAN_ZERO;
        end
        // Then check for overflow
        else if (final_exp >= EXP_INF_NAN) begin // Overflow
            final_exp = EXP_INF_NAN;
            final_mant = MAN_ZERO;
        end
        // Finally handle normal cases
        else begin
            // Adjust exp if rounding caused overflow
            if (rounded_fraction_sum[MAN_BITS]) begin // If rounding caused mantissa overflow
                final_exp = final_exp + 1;
                // Check for overflow after rounding adjustment
                if (final_exp >= EXP_INF_NAN) begin
                    final_exp = EXP_INF_NAN;
                    final_mant = MAN_ZERO;
                end
            end
        end
    end

    assign result_comb = s2_is_special_case_reg ? s2_special_result_word_reg :
                         {s2_res_sign_reg, final_exp, final_mant};

    // Registering stage 3 outputs (final result)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            result <= {TOT_BITS{1'b0}};
        end else begin
            out_valid <= s2_in_valid_reg; // out_valid is the registered version of s2_in_valid_reg
            if (s2_in_valid_reg) begin
                result <= result_comb;
            end
        end
    end

    // Debug: Add test index tracking
    reg [7:0] current_test_index;
    reg first_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_test_index <= 8'd0;
            first_valid <= 1'b1;
        end else if (in_valid) begin
            if (first_valid) begin
                current_test_index <= 8'd0;
                first_valid <= 1'b0;
            end else begin
                current_test_index <= current_test_index + 1;
            end
        end
    end

    // Debug prints for Test 40
    reg [3:0] debug_stage;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            debug_stage <= 4'd0;
        end else if (current_test_index == 8'd40) begin
            if (in_valid) debug_stage <= 4'd1;
            else if (s1_in_valid_reg) debug_stage <= 4'd2;
            else if (s2_in_valid_reg) debug_stage <= 4'd3;
            else if (out_valid) debug_stage <= 4'd4;
            else debug_stage <= 4'd0;
        end
    end

    always @(posedge clk) begin
        if (current_test_index == 8'd40) begin
            case (debug_stage)
                4'd1: begin
                    $display("\n=== Test 40 Pipeline Stage 1 (Input) ===");
                    $display("  Input values:");
                    $display("    a = %h, b = %h", a, b);
                    $display("    a_sign = %b, b_sign = %b", a_sign, b_sign);
                    $display("    a_exp = %h, b_exp = %h", a_exp, b_exp);
                    $display("    a_mant = %h, b_mant = %h", a_mant, b_mant);
                    $display("    a_is_subnormal = %b, b_is_subnormal = %b", a_is_subnormal, b_is_subnormal);
                    $display("    a_is_zero = %b, b_is_zero = %b", a_is_zero, b_is_zero);
                    $display("    a_is_inf = %b, b_is_inf = %b", a_is_inf, b_is_inf);
                    $display("    a_is_nan = %b, b_is_nan = %b", a_is_nan, b_is_nan);
                    $display("=== End Stage 1 ===\n");
                end
                4'd2: begin
                    $display("\n=== Test 40 Pipeline Stage 2 (Calculation) ===");
                    $display("  Stage 1 registered values:");
                    $display("    a_mant_std = %h, b_mant_std = %h", s1_a_mant_std_reg, s1_b_mant_std_reg);
                    $display("    a_is_subnormal = %b, b_is_subnormal = %b", s1_a_is_subnormal_reg, s1_b_is_subnormal_reg);
                    $display("    a_is_zero = %b, b_is_zero = %b", s1_a_is_zero_reg, s1_b_is_zero_reg);
                    $display("  Stage 2 calculations:");
                    $display("    eff_a_exp_s1 = %d, eff_b_exp_s1 = %d", eff_a_exp_s1, eff_b_exp_s1);
                    $display("    s2_res_exp_provisional_comb = %d", s2_res_exp_provisional_comb);
                    $display("    s2_prod_mant_comb = %h", s2_prod_mant_comb);
                    $display("  Special cases:");
                    $display("    s2_res_is_zero = %b", s2_res_is_zero);
                    $display("    s2_res_is_nan = %b", s2_res_is_nan);
                    $display("    s2_res_is_inf = %b", s2_res_is_inf);
                    $display("    s2_res_is_overflow = %b", s2_res_is_overflow);
                    $display("    s2_res_is_underflow = %b", s2_res_is_underflow);
                    $display("=== End Stage 2 ===\n");
                end
                4'd3: begin
                    $display("\n=== Test 40 Pipeline Stage 3 (Normalization) ===");
                    $display("  Stage 2 registered values:");
                    $display("    res_sign = %b", s2_res_sign_reg);
                    $display("    res_exp_provisional = %d", s2_res_exp_provisional_reg);
                    $display("    prod_mant = %h", s2_prod_mant_reg);
                    $display("    is_special_case = %b", s2_is_special_case_reg);
                    $display("    special_result = %h", s2_special_result_word_reg);
                    $display("  Normalization details:");
                    $display("    exp_after_norm_shift = %d", exp_after_norm_shift);
                    $display("    mant_after_norm_shift = %h", mant_after_norm_shift);
                    $display("    left_shift_amount = %d", left_shift_amount);
                    $display("    final_exp = %h", final_exp);
                    $display("    final_mant = %h", final_mant);
                    $display("    rounded_fraction_sum = %h", rounded_fraction_sum);
                    $display("    guard_bit = %b, round_bit = %b, sticky_bit = %b", guard_bit, round_bit, sticky_bit);
                    $display("=== End Stage 3 ===\n");
                end
                4'd4: begin
                    $display("\n=== Test 40 Pipeline Stage 4 (Output) ===");
                    $display("  Final result = %h", result);
                    $display("=== End Stage 4 ===\n");
                end
            endcase
        end
    end

endmodule
