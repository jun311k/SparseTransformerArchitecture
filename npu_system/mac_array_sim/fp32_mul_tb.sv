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
        // Check for NaN
        if ((a[30:23] == 8'hFF && a[22:0] != 0) || 
            (b[30:23] == 8'hFF && b[22:0] != 0)) begin
            category = NAN_CASE;
        end
        // Check for Infinity * zero (results in NaN)
        else if (((a[30:23] == 8'hFF && a[22:0] == 0) || 
                 (b[30:23] == 8'hFF && b[22:0] == 0)) &&
                ((a == 32'h00000000 || a == 32'h80000000) ||
                 (b == 32'h00000000 || b == 32'h80000000))) begin
            category = NAN_CASE;
        end
        // Check for Infinity
        else if ((a[30:23] == 8'hFF && a[22:0] == 0) || 
                 (b[30:23] == 8'hFF && b[22:0] == 0)) begin
            category = INF_CASE;
        end
        // Check for zero
        else if (a == 32'h00000000 || a == 32'h80000000 ||
                 b == 32'h00000000 || b == 32'h80000000) begin
            category = NORMAL_W_ZERO;
        end
        // Check for potential overflow/underflow (IEEE 754 방식)
        else begin
            reg [15:0] exp_sum;
            reg [47:0] mant_prod;
            exp_sum = a[30:23] + b[30:23] - 127;
            //$display("Debug - A: %h, B: %h", a, b);
            //$display("Debug - A exp: %d, B exp: %d", a[30:23], b[30:23]);
            //$display("Debug - Initial exp_sum: %0d (bin: %b, width: %0d)", exp_sum, exp_sum, $bits(exp_sum));

            if (exp_sum[15] == 1'b1 || a[30:23] == 8'h00 || b[30:23] == 8'h00) begin
                category = UNDERFLOW;
            end else if (exp_sum > 254) begin
                category = OVERFLOW;
            end else if (exp_sum == 254) begin
                mant_prod = {1'b1, a[22:0]} * {1'b1, b[22:0]};
                //$display("Debug - mant_prod: %h", mant_prod);
                //$display("Debug - mant_prod[47:46]: %b", mant_prod[47:46]);
                if (mant_prod[47:46] != 2'b00)
                    category = OVERFLOW;
                else
                    category = NORMAL_WO_ZERO;
            end else begin
                category = NORMAL_WO_ZERO;
            end
        end
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
            if (line[0] != "/" && line[0] != "\n" && line[0] != "\r") begin
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
            if (line[0] != "/" && line[0] != "\n" && line[0] != "\r") begin
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
        integer i;
        real power;
        
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
            for (i = 0; i < 23; i = i + 1) begin
                if (temp[22-i]) begin
                    result = result + (1.0 / (1 << (i+126)));  // 126 = 127-1 for subnormal
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
        input [31:0] bits;  // Add bits parameter to check sign
        string result;
        if (value == 0.0) begin
            if (bits[31]) begin  // Check sign bit
                return "-zero";
            end else begin
                return "zero";
            end
        end else if (value == 1.0/0.0) begin
            return "inf";
        end else if (value == -1.0/0.0) begin
            return "-inf";
        end else if (value != value) begin  // NaN check
            if (bits[31]) begin  // Check sign bit
                return "-nan";
            end else begin
                return "nan";
            end
        end else begin
            $sformat(result, "%f", value);
            return result;
        end
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
        case (category)
            NORMAL_WO_ZERO: return "NORMAL_WO_ZERO";
            NORMAL_W_ZERO:  return "NORMAL_W_ZERO";
            INF_CASE:      return "INF_CASE";
            NAN_CASE:      return "NAN_CASE";
            OVERFLOW:      return "OVERFLOW";
            UNDERFLOW:     return "UNDERFLOW";
            DENORMAL:      return "DENORMAL";
            default:       return "UNKNOWN";
        endcase
    endfunction

endmodule
