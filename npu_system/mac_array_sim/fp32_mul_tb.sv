`timescale 1ns/1ps

// FP32 multiplier testbench using limited SystemVerilog features
module fp32_mul_tb;

    // Parameters
    localparam NUM_BASIC_TESTS = 48; // Number of test cases in init_test_cases

    // Test case memory (96 bits: 32-bit padding + 32-bit A + 32-bit B)
    reg [2+32*2-1:0] test_cases_mem [0:999];
    reg [31:0] expected_results_mem [0:999];
    integer num_test_cases;
    integer num_expected_results;
    reg [1:0] test_category;  // Rename to test_category to avoid confusion
    integer first_failed_idx;
    
    // Test statistics
    integer total_tests;
    integer passed_tests;
    integer failed_tests;
    
    // Category statistics
    integer normal_wo_zero_total;
    integer normal_wo_zero_passed;
    integer normal_w_zero_total;
    integer normal_w_zero_passed;
    integer inf_total;
    integer inf_passed;
    integer nan_total;
    integer nan_passed;
    integer overflow_total;
    integer overflow_passed;
    integer underflow_total;
    integer underflow_passed;
    integer denormal_total;
    integer denormal_passed;
    
    // Test case parameters
    reg [31:0] a;
    reg [31:0] b;
    wire [31:0] result;
    reg [31:0] expected_result;
    reg in_valid;
    wire out_valid;
    
    // Clock and reset
    reg clk;
    reg rst_n;
    
    // Test category type
    typedef enum logic [2:0] {
        NORMAL_WO_ZERO = 3'b000,  // Normal numbers without zero
        NORMAL_W_ZERO  = 3'b001,  // Normal numbers with zero
        INF_CASE      = 3'b010,  // Infinity cases
        NAN_CASE      = 3'b011,  // NaN cases
        OVERFLOW      = 3'b100,  // Overflow cases
        UNDERFLOW     = 3'b101,  // Underflow cases
        DENORMAL      = 3'b110   // Denormal numbers
    } test_category_t;

    // Function to determine test category
    function automatic test_category_t get_test_category(
        input [31:0] a,
        input [31:0] b
    );
        // Check for NaN
        if ((a[30:23] == 8'hFF && a[22:0] != 0) || 
            (b[30:23] == 8'hFF && b[22:0] != 0)) begin
            return NAN_CASE;
        end
        
        // Check for Infinity
        if ((a[30:23] == 8'hFF && a[22:0] == 0) || 
            (b[30:23] == 8'hFF && b[22:0] == 0)) begin
            return INF_CASE;
        end
        
        // Check for zero
        if (a == 32'h00000000 || a == 32'h80000000 ||
            b == 32'h00000000 || b == 32'h80000000) begin
            return NORMAL_W_ZERO;
        end
        
        // Check for denormal numbers
        if ((a[30:23] == 8'h00 && a[22:0] != 0) || 
            (b[30:23] == 8'h00 && b[22:0] != 0)) begin
            return DENORMAL;
        end
        
        // Check for potential overflow
        // If sum of exponents would exceed maximum
        if ((a[30:23] > 8'h7F) && (b[30:23] > 8'h7F)) begin
            return OVERFLOW;
        end
        
        // Check for potential underflow
        // If sum of exponents would be too small
        if ((a[30:23] < 8'h7F) && (b[30:23] < 8'h7F) && 
            (a[30:23] + b[30:23] < 8'h7F)) begin
            return UNDERFLOW;
        end
        
        // Normal case without zero
        return NORMAL_WO_ZERO;
    endfunction

    // Instantiate FP32 multiplier
    fp32_mul dut (
        .clk(clk),
        .rst_n(rst_n),
        .a(a),
        .b(b),
        .in_valid(in_valid),
        .out_valid(out_valid),
        .result(result)
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
        $dumpvars(0, fp32_mul_tb.in_valid);
        $dumpvars(0, fp32_mul_tb.out_valid);
        $dumpvars(0, fp32_mul_tb.dut.in_valid);
        $dumpvars(0, fp32_mul_tb.dut.out_valid);
    end
    
    // Initialize test cases
    task init_test_cases;
        integer idx;
        reg [31:0] test_a, test_b, expected;
        reg [1:0] category;  // Local variable for file reading
        integer file;
        reg [8*100:1] line;
        
        // Open input file
        file = $fopen("test_fp32_inputs.txt", "r");
        if (file == 0) begin
            $display("Error: Could not open test_fp32_inputs.txt");
            $finish;
        end

        // Read test cases
        num_test_cases = 0;
        while (!$feof(file)) begin
            // Skip comments and empty lines
            $fgets(line, file);
            if (line[0] != "/" && line[0] != "\n" && line[0] != "\r") begin
                // Parse the line
                if ($sscanf(line, "%d %h %h", category, test_a, test_b) == 3) begin
                    test_cases_mem[num_test_cases] = {test_a, test_b, category};  // Add padding for 96-bit alignment
                    // $display("Debug: test_cases_mem[num_test_cases] = %h", test_cases_mem[num_test_cases]);
                    num_test_cases = num_test_cases + 1;
                    $display("Debug: Test case %0d: Category=%0d A=%h B=%h", 
                            num_test_cases-1, category, test_a, test_b);
                end
            end
        end
        $fclose(file);
        
        // Load expected results
        file = $fopen("test_fp32_expected.txt", "r");
        if (file == 0) begin
            $display("Error: Could not open test_fp32_expected.txt");
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
        a = 32'h00000000;
        b = 32'h00000000;
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
        for (integer i = 0; i < num_test_cases; i = i + 1) begin
            // Apply test case on negedge
            @(negedge clk);
            a = test_cases_mem[i][2+32+31:2+32];
            b = test_cases_mem[i][2+31:2+0];
            test_category = test_cases_mem[i][1:0];
            in_valid = 1;
            
            // Wait for result
            @(negedge clk);
            in_valid = 0;
            wait(out_valid);
            
            // Check result on negedge
            @(negedge clk);
            expected_result = expected_results_mem[i];
            if (result === expected_result) begin
                passed_tests = passed_tests + 1;
                case (test_category)
                    NORMAL_WO_ZERO: begin
                        normal_wo_zero_total = normal_wo_zero_total + 1;
                        normal_wo_zero_passed = normal_wo_zero_passed + 1;
                    end
                    NORMAL_W_ZERO: begin
                        normal_w_zero_total = normal_w_zero_total + 1;
                        normal_w_zero_passed = normal_w_zero_passed + 1;
                    end
                    INF_CASE: begin
                        inf_total = inf_total + 1;
                        inf_passed = inf_passed + 1;
                    end
                    NAN_CASE: begin
                        nan_total = nan_total + 1;
                        nan_passed = nan_passed + 1;
                    end
                    OVERFLOW: begin
                        overflow_total = overflow_total + 1;
                        overflow_passed = overflow_passed + 1;
                    end
                    UNDERFLOW: begin
                        underflow_total = underflow_total + 1;
                        underflow_passed = underflow_passed + 1;
                    end
                    DENORMAL: begin
                        denormal_total = denormal_total + 1;
                        denormal_passed = denormal_passed + 1;
                    end
                endcase
                $display("Test %0d PASSED: A=%h(%s) B=%h(%s) Expected=%h(%s) Got=%h(%s)",
                        i, a, format_real_with_sign(bits_to_real(a), a), 
                        b, format_real_with_sign(bits_to_real(b), b),
                        expected_result, format_real_with_sign(bits_to_real(expected_result), expected_result),
                        result, format_real_with_sign(bits_to_real(result), result));
            end else begin
                failed_tests = failed_tests + 1;
                if (first_failed_idx == -1) begin
                    first_failed_idx = i;  // Record first failed test case
                end
                $display("\nTest %0d FAILED:", i);
                $display("  A = %h (%s)", a, format_real_with_sign(bits_to_real(a), a));
                $display("  B = %h (%s)", b, format_real_with_sign(bits_to_real(b), b));
                $display("  Expected = %h (%s)", expected_result, format_real_with_sign(bits_to_real(expected_result), expected_result));
                $display("  Got = %h (%s)", result, format_real_with_sign(bits_to_real(result), result));
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
            $display("  Got = %h (%s)", result, format_real_with_sign(bits_to_real(result), result));
        end
        
        // End simulation
        #100;
        $finish;
    end

endmodule
