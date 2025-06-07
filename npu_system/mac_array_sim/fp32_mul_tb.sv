`timescale 1ns/1ps

// FP32 multiplier testbench using limited SystemVerilog features
module fp32_mul_tb;

    // Parameters
    localparam MAX_TEST_CASES = 1000;

    // Test case memory (96 bits: 32-bit A + 32-bit B + 32-bit category)
    logic [32*3-1:0] test_cases_mem [0:MAX_TEST_CASES-1];
    logic [31:0] expected_results_mem [0:MAX_TEST_CASES-1];
    logic [31:0] actual_results_mem [0:MAX_TEST_CASES-1];  // Add memory for actual results
    integer num_test_cases;
    integer num_expected_results;
    integer first_failed_idx;
    integer test_index;  // Add test_index for waveform viewing
    integer current_test_idx;  // Add current_test_idx for waveform viewing
    logic [31:0] wait_counter;
    
    // Test signals
    logic [31:0] read_a;  // Current test input A
    logic [31:0] read_b;  // Current test input B
    logic [31:0] category;  // Current test category
    logic [31:0] dut_a;    // DUT input A
    logic [31:0] dut_b;    // DUT input B
    logic [31:0] dut_result;  // DUT output result
    logic in_valid;
    logic out_valid;
    logic [31:0] read_result;
    
    // Test category definitions
    localparam NORMAL_WO_ZERO = 32'd0;  // Normal multiplication without zero
    localparam NORMAL_W_ZERO  = 32'd1;  // Normal multiplication with zero
    localparam INF_CASE      = 32'd2;  // Infinity cases
    localparam NAN_CASE      = 32'd3;  // NaN cases
    localparam OVERFLOW      = 32'd4;  // Overflow cases
    localparam UNDERFLOW     = 32'd5;  // Underflow cases
    localparam DENORMAL      = 32'd6;  // Denormal cases
    localparam INF_NAN_COMB  = 32'd7;  // Infinity and NaN combinations
    localparam EDGE_THRESHOLD = 32'd8; // Edge cases near thresholds
    localparam ROUNDING      = 32'd9;  // Rounding cases
    localparam SUBNORMAL_RESULT = 32'd10; // Subnormal results
    localparam MAX_SHIFT     = 32'd11; // Maximum normalization shifts
    
    // Test category tracking
    integer normal_wo_zero_total = 0;
    integer normal_wo_zero_passed = 0;
    integer normal_w_zero_total = 0;
    integer normal_w_zero_passed = 0;
    integer inf_total = 0;
    integer inf_passed = 0;
    integer nan_total = 0;
    integer nan_passed = 0;
    integer overflow_total = 0;
    integer overflow_passed = 0;
    integer underflow_total = 0;
    integer underflow_passed = 0;
    integer denormal_total = 0;
    integer denormal_passed = 0;
    integer inf_nan_comb_total = 0;
    integer inf_nan_comb_passed = 0;
    integer edge_threshold_total = 0;
    integer edge_threshold_passed = 0;
    integer rounding_total = 0;
    integer rounding_passed = 0;
    integer subnormal_result_total = 0;
    integer subnormal_result_passed = 0;
    integer max_shift_total = 0;
    integer max_shift_passed = 0;
    
    // Test statistics
    integer total_tests;
    integer passed_tests;
    integer failed_tests;
    
    // Test case parameters
    reg [31:0] expected_result;

    
    // Clock and reset
    reg clk;
    reg rst_n;
    
    // Task to determine test category
    task get_test_category(
        input [31:0] a,
        input [31:0] b,
        output [31:0] category
    );
        // Debug prints for input values
        $display("\n=== Debug: Test Case Classification ===");
        $display("Input A: %h (exp: %h, mant: %h)", a, a[30:23], a[22:0]);
        $display("Input B: %h (exp: %h, mant: %h)", b, b[30:23], b[22:0]);

        // Check for Infinity and NaN combinations
        if ((a[30:23] == 8'hFF && b[30:23] == 8'hFF) ||
            ((a[30:23] == 8'hFF && a[22:0] != 0 && b[30:23] == 8'hFF && b[22:0] != 0))) begin
            category = INF_NAN_COMB;
            $display("Category: INF_NAN_COMB (Infinity and NaN combination)");
        end
        // Check for NaN
        else if ((a[30:23] == 8'hFF && a[22:0] != 0) || 
                 (b[30:23] == 8'hFF && b[22:0] != 0)) begin
            category = NAN_CASE;
            $display("Category: NAN_CASE (NaN input)");
        end
        // Check for Infinity * zero (results in NaN)
        else if (((a[30:23] == 8'hFF && a[22:0] == 0) || 
                 (b[30:23] == 8'hFF && b[22:0] == 0)) &&
                ((a == 32'h00000000 || a == 32'h80000000) ||
                 (b == 32'h00000000 || b == 32'h80000000))) begin
            category = NAN_CASE;
            $display("Category: NAN_CASE (Infinity * zero)");
        end
        // Check for Infinity
        else if ((a[30:23] == 8'hFF && a[22:0] == 0) || 
                 (b[30:23] == 8'hFF && b[22:0] == 0)) begin
            category = INF_CASE;
            $display("Category: INF_CASE (Infinity input)");
        end
        // Check for zero
        else if (a == 32'h00000000 || a == 32'h80000000 ||
                 b == 32'h00000000 || b == 32'h80000000) begin
            // If one input is subnormal and other is zero, classify as SUBNORMAL_RESULT
            reg a_is_denormal, b_is_denormal;
            a_is_denormal = (a[30:23] == 8'h00 && a[22:0] != 0);
            b_is_denormal = (b[30:23] == 8'h00 && b[22:0] != 0);

            if (a_is_denormal || b_is_denormal) begin
                category = SUBNORMAL_RESULT;
                $display("Category: SUBNORMAL_RESULT (Subnormal * Zero)");
            end else begin
                category = NORMAL_W_ZERO;
                $display("Category: NORMAL_W_ZERO (Normal * Zero)");
            end
        end
        // Check for denormal numbers and calculate actual exponents
        else begin
            reg [15:0] exp_sum;
            reg [47:0] mant_prod;
            reg [7:0] a_actual_exp, b_actual_exp;
            reg a_is_denormal, b_is_denormal;

            // Check for denormal numbers
            a_is_denormal = (a[30:23] == 8'h00 && a[22:0] != 0) || 
                           (a[30:23] == 8'h01 && a[22:0] != 0 && a[22:0] < 23'h800000);
            b_is_denormal = (b[30:23] == 8'h00 && b[22:0] != 0) || 
                           (b[30:23] == 8'h01 && b[22:0] != 0 && b[22:0] < 23'h800000);

            // Calculate actual exponents considering denormal numbers
            a_actual_exp = a_is_denormal ? 8'h01 : a[30:23];
            b_actual_exp = b_is_denormal ? 8'h01 : b[30:23];
            exp_sum = a_actual_exp + b_actual_exp - 127;

            $display("Actual exponents - A: %h, B: %h", a_actual_exp, b_actual_exp);
            $display("Exponent sum: %d", exp_sum);
            $display("Denormal flags - A: %b, B: %b", a_is_denormal, b_is_denormal);

            // If both inputs are denormal, result will be underflow
            if (a_is_denormal && b_is_denormal) begin
                category = UNDERFLOW;
                $display("Category: UNDERFLOW (Both inputs are denormal, result will be zero)");
            end
            // If one input is denormal and result exponent is in denormal range
            else if (a_is_denormal || b_is_denormal) begin
                if (exp_sum < 0) begin
                    category = UNDERFLOW;
                    $display("Category: UNDERFLOW (Denormal input with negative result exponent)");
                end else if (a_is_denormal || b_is_denormal) begin
                    category = SUBNORMAL_RESULT;
                    $display("Category: SUBNORMAL_RESULT (Subnormal operation)");
                end else begin
                    category = DENORMAL;
                    $display("Category: DENORMAL (Denormal input with non-negative result exponent)");
                end
            end
            // Check for underflow from normal inputs
            else if (exp_sum[15] == 1'b1) begin
                category = UNDERFLOW;
                $display("Category: UNDERFLOW (Normal inputs with negative result exponent)");
            end
            // Check for overflow
            else if (exp_sum > 254) begin
                category = OVERFLOW;
                $display("Category: OVERFLOW (Result exponent too large)");
            end
            // Check for edge case near overflow
            else if (exp_sum == 254) begin
                mant_prod = {1'b1, a[22:0]} * {1'b1, b[22:0]};
                if (mant_prod[47:46] != 2'b00) begin
                    category = OVERFLOW;
                    $display("Category: OVERFLOW (Edge case with mantissa overflow)");
                end else begin
                    category = NORMAL_WO_ZERO;
                    $display("Category: NORMAL_WO_ZERO (Edge case without mantissa overflow)");
                end
            end
            // Check other cases
            else begin
                // Check for edge cases near thresholds
                if (a == 32'h00800000 || b == 32'h00800000 ||  // min normal
                    (a == 32'h7f7fffff && b == 32'h3f7fffff) ||  // max normal * 0.9999999
                    (b == 32'h7f7fffff && a == 32'h3f7fffff)) begin
                    category = EDGE_THRESHOLD;
                    $display("Category: EDGE_THRESHOLD (Edge case near threshold)");
                end
                // Check for rounding cases - only for specific test cases
                else if ((a == 32'h3F800001 && b == 32'h3F800001) ||  // 1.0000001 * 1.0000001
                         (a == 32'h3F800002 && b == 32'h3F800002) ||  // 1.0000002 * 1.0000002
                         (a == 32'h3F800003 && b == 32'h3F800003) ||  // 1.0000003 * 1.0000003
                         (a == 32'h3F800004 && b == 32'h3F800004)) begin  // 1.0000004 * 1.0000004
                    category = ROUNDING;
                    $display("Category: ROUNDING (Specific rounding test case)");
                end
                // Check for subnormal results
                else if ((a[30:23] == 8'h7F && b[30:23] == 8'h00) ||
                         (a[30:23] == 8'h00 && b[30:23] == 8'h7F)) begin
                    category = SUBNORMAL_RESULT;
                    $display("Category: SUBNORMAL_RESULT (Subnormal result)");
                end
                // Check for maximum shift cases
                else if ((a[30:23] == 8'h7F && b[30:23] == 8'h00) ||
                         (a[30:23] == 8'h00 && b[30:23] == 8'h00)) begin
                    category = MAX_SHIFT;
                    $display("Category: MAX_SHIFT (Maximum shift case)");
                end
                else begin
                    category = NORMAL_WO_ZERO;
                    $display("Category: NORMAL_WO_ZERO (Default case)");
                end
            end
        end
        $display("=== End Debug ===\n");
    endtask

    // DUT instance
    fp32_mul dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .a(dut_a),
        .b(dut_b),
        .out_valid(out_valid),
        .result(dut_result)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz clock
    end
    
    // VCD file generation
    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, fp32_mul_tb);
    end
    
    // Initialize test cases
    task init_test_cases;
        integer idx;
        reg [31:0] test_a, test_b, expected;
        reg [31:0] file_category;  // Add declaration for file_category
        integer file;
        reg [8*100:1] line;
        
        // Open input file
        file = $fopen("fp32_inputs.txt", "r");
        if (file == 0) begin
            $display("Error: Could not open fp32_inputs.txt");
            $finish;
        end

        // Read test cases
        num_test_cases = 0;
        test_index = 0;  // Initialize test_index
        while (!$feof(file)) begin
            // Skip comments and empty lines
            $fgets(line, file);
            // $display("contents of testcase %d : %s", test_index, line);
            if (line != "" && line[0] != "/" && line[0] != "\n" && line[0] != "\r") begin
                // Parse the line
                if ($sscanf(line, "%d %h %h", file_category, test_a, test_b) == 3) begin
                    get_test_category(test_a, test_b, category);  // Calculate category using task
                    
                    // Verify category matches
                    if (category !== file_category) begin
                        $display("Error: Category mismatch for test case %0d:", test_index);
                        $display("  File category: %0d (%s)", file_category, get_category_name(file_category));
                        $display("  Calculated category: %0d (%s)", category, get_category_name(category));
                        $display("  A: %h", test_a);
                        $display("  B: %h", test_b);
                        $finish;
                    end
                    
                    test_cases_mem[num_test_cases] = {category, test_a, test_b};  // Store category in upper 32 bits
                    num_test_cases = num_test_cases + 1;
                    test_index = test_index + 1;  // Increment test_index
                    $display("Debug: Test case %0d: Category=%0d (%s) A=%h B=%h", 
                            test_index-1, category, get_category_name(category), test_a, test_b);
                end
            end
        end
        $fclose(file);
        
        // Load expected results
        file = $fopen("fp32_expected.txt", "r");
        if (file == 0) begin
            $display("Error: Could not open fp32_expected.txt");
            $finish;
        end

        num_expected_results = 0;
        while (!$feof(file)) begin
            // Skip comments and empty lines
            $fgets(line, file);
            if (line != "" && line[0] != "/" && line[0] != "\n" && line[0] != "\r") begin
                // Parse the line
                if ($sscanf(line, "%h", expected) == 1) begin
                    expected_results_mem[num_expected_results] = expected;
                    num_expected_results = num_expected_results + 1;
                    $display("Debug: Expected result %0d: %h", 
                            num_expected_results-1, expected);
                end
            end
        end
        $fclose(file);
        
        // Verify we have matching numbers of test cases and expected results
        if (num_test_cases != num_expected_results) begin
            $display("Error: Number of test cases (%0d) does not match number of expected results (%0d)",
                    num_test_cases, num_expected_results);
            $finish;
        end
        
        $display("Loaded %0d test cases from files", num_test_cases);
    endtask
    
    // Run random tests (will be skipped for now as we only have fixed test inputs)
    task run_random_tests;
        input int num_tests;
        $display("\nSkipping random tests as they are not supported with pre-generated expected results.");
    endtask
    
    // Function to convert bits to real
    function real bits_to_real;
        input [31:0] bits;
        real result;
        reg [31:0] temp;
        integer i, j;
        real power, base, bit_value;
        
        // Handle special cases
        if (bits[30:23] == 8'hFF) begin
            if (bits[22:0] == 0) begin
                // Infinity
                if (bits[31]) begin
                    return -1.0/0.0;  // -Inf
                end else begin
                    return 1.0/0.0;   // +Inf
                end
            end else begin
                // NaN
                if (bits[31]) begin
                    return -0.0/0.0;  // -NaN
                end else begin
                    return 0.0/0.0;   // +NaN
                end
            end
        end
        
        // Handle zero
        if (bits[30:0] == 0) begin
            if (bits[31]) begin
                return -0.0;  // -0
            end else begin
                return 0.0;   // +0
            end
        end
        
        // Handle subnormal numbers
        if (bits[30:23] == 8'h00) begin
            temp = bits;
            result = 0.0;
            power = 1.0;
            
            // Handle sign
            if (temp[31]) begin
                temp[31] = 0;
                power = -1.0;
            end
            
            // Handle mantissa for subnormal
            result = 0.0;
            base = 1.0;
            // Calculate 2^-126 once
            for (j = 0; j < 126; j = j + 1) begin
                base = base / 2.0;
            end
            
            for (i = 0; i < 23; i = i + 1) begin
                if (temp[22-i]) begin
                    // For subnormal numbers, each bit represents 2^-126 * 2^-i
                    bit_value = base / (1 << i);
                    result = result + bit_value;
                end
            end
            
            return result * power;
        end
        
        // Normal number
        temp = bits;
        result = 0.0;
        power = 1.0;
        
        // Handle sign
        if (temp[31]) begin
            temp[31] = 0;
            power = -1.0;
        end
        
        // Handle exponent
        for (i = 0; i < 8; i = i + 1) begin
            if (temp[30-i]) begin
                result = result + (1 << (7-i));
            end
        end
        result = result - 127.0;
        power = power * (2.0 ** result);
        
        // Handle mantissa
        result = 1.0;
        for (i = 0; i < 23; i = i + 1) begin
            if (temp[22-i]) begin
                result = result + (1.0 / (1 << (i+1)));
            end
        end
        
        return result * power;
    endfunction

    // Function to format real number with sign
    function string format_real_with_sign;
        input real value;
        input [31:0] bits;
        string result;
        real abs_value;
        
        // Handle special cases
        if (bits[30:23] == 8'hFF) begin
            if (bits[22:0] == 0) begin
                // Infinity
                if (bits[31]) begin
                    return "-inf";
                end else begin
                    return "+inf";
                end
            end else begin
                // NaN
                if (bits[31]) begin
                    return "-nan";
                end else begin
                    return "+nan";
                end
            end
        end
        
        // Handle zero
        if (bits[30:0] == 0) begin
            if (bits[31]) begin
                return "-0.000000e+00";
            end else begin
                return "+0.000000e+00";
            end
        end
        
        // Handle normal numbers
        abs_value = (value < 0) ? -value : value;
        
        // Use scientific notation with 6 decimal places
        if (value < 0) begin
            $sformat(result, "-%.6e", abs_value);
        end else begin
            $sformat(result, "+%.6e", abs_value);
        end
        
        return result;
    endfunction

    // Main test sequence
    initial begin
        // Initialize
        rst_n = 0;
        dut_a = 32'h00000000;
        dut_b = 32'h00000000;
        total_tests = 0;
        passed_tests = 0;
        failed_tests = 0;
        
        // Initialize category statistics
        normal_wo_zero_total = 0;
        normal_wo_zero_passed = 0;
        normal_w_zero_total = 0;
        normal_w_zero_passed = 0;
        inf_total = 0;
        inf_passed = 0;
        nan_total = 0;
        nan_passed = 0;
        overflow_total = 0;
        overflow_passed = 0;
        underflow_total = 0;
        underflow_passed = 0;
        denormal_total = 0;
        denormal_passed = 0;
        inf_nan_comb_total = 0;
        inf_nan_comb_passed = 0;
        edge_threshold_total = 0;
        edge_threshold_passed = 0;
        rounding_total = 0;
        rounding_passed = 0;
        subnormal_result_total = 0;
        subnormal_result_passed = 0;
        max_shift_total = 0;
        max_shift_passed = 0;
        in_valid = 0;
        first_failed_idx = -1;  // Initialize first failed index

        $display("=== FP32 Multiplier Testbench ===");
        
        // Initialize test cases
        init_test_cases();
        
        // Release reset
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Run all test cases
        $display("\n=== Running Tests ===");
        current_test_idx = 0;  // Initialize current_test_idx
        for (integer i = 0; i < num_test_cases; i = i + 1) begin
            current_test_idx = i;  // Update current_test_idx for waveform
            // Get test case data
            read_a = test_cases_mem[i][31:0];
            read_b = test_cases_mem[i][63:32];
            category = test_cases_mem[i][95:64];
            
            // Apply test case on negedge
            @(negedge clk);
            dut_a = read_a;  // Assign to DUT input
            dut_b = read_b;  // Assign to DUT input
            in_valid = 1'b1;
            // $display("Debug: Test %0d - Applied inputs: A=%h, B=%h, in_valid=%d, %t", i, dut_a, dut_b, in_valid, $time);
            
            // Wait for result
            @(negedge clk);
            in_valid = 1'b0;
            wait_counter = 0;
            // $display("Debug: Test %0d - Waiting for out_valid...after in_valid=%d, %t", i, in_valid, $time);
            
            @(posedge clk);
            // $display("Debug: Test %0d - next_clock", i);

            // Wait for out_valid with timeout
            while (!out_valid && wait_counter < 10) begin
                @(negedge clk);
                wait_counter += 1;
            end
            
            if (!out_valid) begin
                $display("Error: Test %0d - out_valid did not assert within timeout", i);
                $finish;
            end
            // $display("Debug: Test %0d - out_valid asserted", i);
            
            // Wait for one more clock edge to ensure result is stable
            @(posedge clk);
            read_result = dut_result;
            actual_results_mem[i] = read_result;  // Store the actual result
            // $display("Debug: Test %0d - Captured result: %h", i, read_result);

            // Get expected result
            expected_result = expected_results_mem[i];
            
            // Update category statistics first, regardless of pass/fail
            case (category)
                NORMAL_WO_ZERO: normal_wo_zero_total = normal_wo_zero_total + 1;
                NORMAL_W_ZERO: normal_w_zero_total = normal_w_zero_total + 1;
                INF_CASE: inf_total = inf_total + 1;
                NAN_CASE: nan_total = nan_total + 1;
                OVERFLOW: overflow_total = overflow_total + 1;
                UNDERFLOW: underflow_total = underflow_total + 1;
                DENORMAL: denormal_total = denormal_total + 1;
                INF_NAN_COMB: inf_nan_comb_total = inf_nan_comb_total + 1;
                EDGE_THRESHOLD: edge_threshold_total = edge_threshold_total + 1;
                ROUNDING: rounding_total = rounding_total + 1;
                SUBNORMAL_RESULT: subnormal_result_total = subnormal_result_total + 1;
                MAX_SHIFT: max_shift_total = max_shift_total + 1;
                default: $display("Warning: Unknown test category %0d", category);
            endcase

            if (read_result === expected_result) begin
                passed_tests = passed_tests + 1;
                case (category)
                    NORMAL_WO_ZERO: normal_wo_zero_passed = normal_wo_zero_passed + 1;
                    NORMAL_W_ZERO: normal_w_zero_passed = normal_w_zero_passed + 1;
                    INF_CASE: inf_passed = inf_passed + 1;
                    NAN_CASE: nan_passed = nan_passed + 1;
                    OVERFLOW: overflow_passed = overflow_passed + 1;
                    UNDERFLOW: underflow_passed = underflow_passed + 1;
                    DENORMAL: denormal_passed = denormal_passed + 1;
                    INF_NAN_COMB: inf_nan_comb_passed = inf_nan_comb_passed + 1;
                    EDGE_THRESHOLD: edge_threshold_passed = edge_threshold_passed + 1;
                    ROUNDING: rounding_passed = rounding_passed + 1;
                    SUBNORMAL_RESULT: subnormal_result_passed = subnormal_result_passed + 1;
                    MAX_SHIFT: max_shift_passed = max_shift_passed + 1;
                endcase
                $display("Test %0d PASSED [Category: %s]: A=%h(%s) B=%h(%s) Expected=%h(%s) Got=%h(%s)",
                        i, get_category_name(category),
                        read_a, format_real_with_sign(bits_to_real(read_a), read_a), 
                        read_b, format_real_with_sign(bits_to_real(read_b), read_b),
                        expected_result, format_real_with_sign(bits_to_real(expected_result), expected_result),
                        actual_results_mem[i], format_real_with_sign(bits_to_real(actual_results_mem[i]), actual_results_mem[i]));
            end else begin
                failed_tests = failed_tests + 1;
                if (first_failed_idx == -1) begin
                    first_failed_idx = i;  // Record first failed test case
                end
                $display("\nTest %0d FAILED [Category: %s]:", i, get_category_name(category));
                $display("  A = %h (%s)", read_a, format_real_with_sign(bits_to_real(read_a), read_a));
                $display("  B = %h (%s)", read_b, format_real_with_sign(bits_to_real(read_b), read_b));
                $display("  Expected = %h (%s)", expected_result, format_real_with_sign(bits_to_real(expected_result), expected_result));
                $display("  Got = %h (%s)", actual_results_mem[i], format_real_with_sign(bits_to_real(actual_results_mem[i]), actual_results_mem[i]));
            end
            total_tests = total_tests + 1;
            
            // Wait a few cycles between tests
            repeat(2) @(posedge clk);
        end
        
        // Print test summary
        $display("\n=== Test Summary ===");
        $display("Total tests: %0d", total_tests);
        $display("Passed: %0d", passed_tests);
        $display("Failed: %0d", failed_tests);
        
        // Print detailed category results
        $display("\n=== Category Results ===");
        $display("Normal cases without zero: %0d/%0d passed", normal_wo_zero_passed, normal_wo_zero_total);
        $display("Normal cases with zero: %0d/%0d passed", normal_w_zero_passed, normal_w_zero_total);
        $display("Infinity cases: %0d/%0d passed", inf_passed, inf_total);
        $display("NaN cases: %0d/%0d passed", nan_passed, nan_total);
        $display("Overflow cases: %0d/%0d passed", overflow_passed, overflow_total);
        $display("Underflow cases: %0d/%0d passed", underflow_passed, underflow_total);
        $display("Denormal cases: %0d/%0d passed", denormal_passed, denormal_total);
        $display("Infinity and NaN combinations: %0d/%0d passed", inf_nan_comb_passed, inf_nan_comb_total);
        $display("Edge cases near thresholds: %0d/%0d passed", edge_threshold_passed, edge_threshold_total);
        $display("Rounding cases: %0d/%0d passed", rounding_passed, rounding_total);
        $display("Subnormal results: %0d/%0d passed", subnormal_result_passed, subnormal_result_total);
        $display("Maximum normalization shifts: %0d/%0d passed", max_shift_passed, max_shift_total);
        
        // Print first failed test case again
        if (first_failed_idx != -1) begin
            $display("\n=== First Failed Test Case ===");
            $display("Test %0d:", first_failed_idx);
            $display("  A = %h (%s)", test_cases_mem[first_failed_idx][63:32], 
                    format_real_with_sign(bits_to_real(test_cases_mem[first_failed_idx][63:32]), test_cases_mem[first_failed_idx][63:32]));
            $display("  B = %h (%s)", test_cases_mem[first_failed_idx][31:0], 
                    format_real_with_sign(bits_to_real(test_cases_mem[first_failed_idx][31:0]), test_cases_mem[first_failed_idx][31:0]));
            $display("  Expected = %h (%s)", expected_results_mem[first_failed_idx], 
                    format_real_with_sign(bits_to_real(expected_results_mem[first_failed_idx]), expected_results_mem[first_failed_idx]));
            $display("  Got = %h (%s)", actual_results_mem[first_failed_idx], 
                    format_real_with_sign(bits_to_real(actual_results_mem[first_failed_idx]), actual_results_mem[first_failed_idx]));
        end
        
        // End simulation
        #100;
        $finish;
    end

    // Function to get category name
    function string get_category_name;
        input [31:0] category;
        begin
            case (category)
                NORMAL_WO_ZERO: get_category_name = "NORMAL_WO_ZERO";
                NORMAL_W_ZERO:  get_category_name = "NORMAL_W_ZERO";
                INF_CASE:      get_category_name = "INF_CASE";
                NAN_CASE:      get_category_name = "NAN_CASE";
                OVERFLOW:      get_category_name = "OVERFLOW";
                UNDERFLOW:     get_category_name = "UNDERFLOW";
                DENORMAL:      get_category_name = "DENORMAL";
                INF_NAN_COMB:  get_category_name = "INF_NAN_COMB";
                EDGE_THRESHOLD: get_category_name = "EDGE_THRESHOLD";
                ROUNDING:      get_category_name = "ROUNDING";
                SUBNORMAL_RESULT: get_category_name = "SUBNORMAL_RESULT";
                MAX_SHIFT:     get_category_name = "MAX_SHIFT";
                default:       get_category_name = "UNKNOWN";
            endcase
        end
    endfunction

endmodule
