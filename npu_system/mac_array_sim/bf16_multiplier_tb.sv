`timescale 1ns/1ps

// BF16 multiplier testbench using limited SystemVerilog features
module bf16_multiplier_tb;
    // Clock and reset
    logic clk;
    logic rst_n;
    
    // Test vectors
    logic [15:0] a;
    logic [15:0] b;
    logic [15:0] result;
    logic in_valid;
    logic out_valid;
    
    // Expected results
    logic [15:0] expected_result;
    
    // Test statistics
    int total_tests;
    int passed_tests;
    int failed_tests;
    
    // Test category counters
    int normal_passed;
    int normal_total;
    int special_passed;
    int special_total;
    int edge_passed;
    int edge_total;
    
    // Pipeline tracking
    logic [15:0] pipeline_a[0:2];
    logic [15:0] pipeline_b[0:2];
    logic [15:0] pipeline_result[0:2];
    
    // Instantiate BF16 multiplier
    bf16_multiplier dut (
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
        $dumpvars(0, bf16_multiplier_tb);
        $dumpvars(0, bf16_multiplier_tb.in_valid);
        $dumpvars(0, bf16_multiplier_tb.out_valid);
        $dumpvars(0, bf16_multiplier_tb.dut.in_valid);
        $dumpvars(0, bf16_multiplier_tb.dut.out_valid);
    end
    
    // Run test case
    task run_test_case;
        input [15:0] test_a;
        input [15:0] test_b;
        input [15:0] expected;
        input [1:0] category;  // 0: normal, 1: special, 2: edge
        
        $display("\nRunning test case: A=%h, B=%h", test_a, test_b);
        @(negedge clk); // Change signals at negedge
        a = test_a;
        b = test_b;
        expected_result = expected;
        in_valid = 1'b1;
        
        // Update pipeline tracking
        pipeline_a[0] = a;
        pipeline_b[0] = b;
        
        // Wait for pipeline (3 cycles for the actual implementation)
        repeat(3) @(posedge clk);
        @(negedge clk); // Deassert in_valid at negedge
        in_valid = 1'b0;
        
        // Wait for 10 clock cycles before next test
        repeat(10) @(posedge clk);
        
        // Update category counters before checking result
        case (category)
            2'b00: normal_total++;
            2'b01: special_total++;
            2'b10: edge_total++;
        endcase
        
        // Check result
        if (result === expected_result) begin
            $display("PASSED: A=%h, B=%h", a, b);
            $display("  Time: %t", $time);
            passed_tests++;
            
            // Update category counters
            case (category)
                2'b00: normal_passed++;
                2'b01: special_passed++;
                2'b10: edge_passed++;
            endcase
        end else begin
            $display("FAILED: A=%h, B=%h", a, b);
            $display("  Time: %t", $time);
            $display("  Input A: %h", a);
            $display("  Input B: %h", b);
            $display("  Expected: %h", expected_result);
            $display("  Got: %h", result);
            failed_tests++;
        end
        total_tests++;
    endtask
    
    // Initialize test cases
    task init_test_cases;
        // Normal test cases
        run_test_case(16'h3F80, 16'h4000, 16'h4000, 2'b00);  // 1.0 * 2.0
        run_test_case(16'hBF80, 16'h4000, 16'hC000, 2'b00);  // -1.0 * 2.0
        run_test_case(16'h3F00, 16'h3F00, 16'h3E80, 2'b00);  // 0.5 * 0.5
        
        // Special cases
        run_test_case(16'h0000, 16'h3F80, 16'h0000, 2'b01);  // 0.0 * 1.0
        run_test_case(16'h7F80, 16'h3F80, 16'h7F80, 2'b01);  // inf * 1.0
        run_test_case(16'h7FC0, 16'h3F80, 16'h7FC0, 2'b01);  // NaN * 1.0
        
        // Edge cases
        run_test_case(16'h7F00, 16'h7F00, 16'h7F80, 2'b10);  // 1e38 * 1e38
        run_test_case(16'h0080, 16'h0080, 16'h0000, 2'b10);  // 1e-38 * 1e-38
        run_test_case(16'h0000, 16'h7F80, 16'h7FC0, 2'b10);  // 0 * inf
    endtask
    
    // Run random tests
    task run_random_tests;
        input int num_tests;
        int i;
        reg [31:0] rand_val;
        
        $display("\nRunning %0d random tests", num_tests);
        
        for (i = 0; i < num_tests; i++) begin
            // Generate random inputs using $random
            rand_val = $random;
            a = rand_val[15:0];
            rand_val = $random;
            b = rand_val[15:0];
            
            // Calculate expected result
            expected_result = calculate_bf16_multiply(a, b);
            
            // Update pipeline tracking
            pipeline_a[0] = a;
            pipeline_b[0] = b;
            
            // Wait for pipeline (3 cycles for the actual implementation)
            repeat(3) @(posedge clk);
            
            // Update counters before checking result
            normal_total++;
            
            // Check result
            if (result === expected_result) begin
                passed_tests++;
                normal_passed++;
            end else begin
                $display("Random test %0d failed:", i);
                $display("  Time: %t", $time);
                $display("  Input A: %h", a);
                $display("  Input B: %h", b);
                $display("  Expected: %h", expected_result);
                $display("  Got: %h", result);
                failed_tests++;
            end
            total_tests++;
        end
    endtask
    
    // Pipeline test
    task run_pipeline_test;
        $display("\n=== Running Pipeline Test ===");
        a = 16'h3F80; // 1.0
        b = 16'h4000; // 2.0
        in_valid = 1'b1;
        repeat(3) @(posedge clk);
        in_valid = 1'b0;
        repeat(10) @(posedge clk);
        if (result === 16'h4000) begin
            $display("Pipeline Test PASSED");
        end else begin
            $display("Pipeline Test FAILED");
        end
    endtask
    
    // Main test sequence
    initial begin
        // Initialize
        rst_n = 0;
        a = 16'h0000;
        b = 16'h0000;
        total_tests = 0;
        passed_tests = 0;
        failed_tests = 0;
        normal_passed = 0;
        normal_total = 0;
        special_passed = 0;
        special_total = 0;
        edge_passed = 0;
        edge_total = 0;
        in_valid = 0; // Set initial value of in_valid to 0
        
        // Release reset
        repeat(4) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);
        
        // Run basic test cases
        $display("\n=== Running Basic Tests ===");
        init_test_cases();
        
        // Print basic test results
        $display("\n=== Basic Test Results ===");
        $display("Normal Tests: %0d/%0d passed", normal_passed, normal_total);
        $display("Special Tests: %0d/%0d passed", special_passed, special_total);
        $display("Edge Tests: %0d/%0d passed", edge_passed, edge_total);
        
        // Check if basic tests passed
        if (normal_passed == normal_total && 
            special_passed == special_total && 
            edge_passed == edge_total) begin
            // Run pipeline test
            run_pipeline_test();
            // Run random tests only if all basic tests passed
            $display("\n=== Running Random Tests ===");
            run_random_tests(500);
        end else begin
            $display("\n=== Skipping Random Tests - Basic Tests Failed ===");
        end
        
        // Print final test summary
        $display("\n=== Final Test Summary ===");
        $display("Basic Tests:");
        $display("  Normal Tests: %0d/%0d passed", normal_passed, normal_total);
        $display("  Special Tests: %0d/%0d passed", special_passed, special_total);
        $display("  Edge Tests: %0d/%0d passed", edge_passed, edge_total);
        $display("Total Tests: %0d", total_tests);
        $display("Total Passed: %0d", passed_tests);
        $display("Total Failed: %0d", failed_tests);
        
        // End simulation
        #100;
        $finish;
    end
    
    // Function to calculate BF16 multiplication
    function [15:0] calculate_bf16_multiply;
        input [15:0] a, b;
        logic a_sign, b_sign;
        logic [7:0] a_exp, b_exp;
        logic [6:0] a_mant, b_mant;
        logic result_sign;
        logic [7:0] result_exp;
        logic [6:0] result_mant;
        logic [13:0] mult_result;
        
        // Extract components
        a_sign = a[15];
        b_sign = b[15];
        a_exp = a[14:7];
        b_exp = b[14:7];
        a_mant = a[6:0];
        b_mant = b[6:0];
        
        // Handle special cases
        if (a_exp == 8'hFF || b_exp == 8'hFF) begin
            if (a_mant != 0 || b_mant != 0) begin
                // NaN
                calculate_bf16_multiply = {1'b0, 8'hFF, 7'h40};
            end else if ((a_exp == 8'hFF && b_mant == 0) || 
                       (b_exp == 8'hFF && a_mant == 0)) begin
                // Infinity
                calculate_bf16_multiply = {a_sign ^ b_sign, 8'hFF, 7'h00};
            end else begin
                // 0 * inf = NaN
                calculate_bf16_multiply = {1'b0, 8'hFF, 7'h40};
            end
        end else if (a_exp == 0 || b_exp == 0) begin
            // Zero
            calculate_bf16_multiply = {a_sign ^ b_sign, 8'h00, 7'h00};
        end else begin
            // Normal multiplication
            result_sign = a_sign ^ b_sign;
            result_exp = a_exp + b_exp - 8'd127;
            mult_result = {1'b1, a_mant} * {1'b1, b_mant};
            
            // Normalize
            if (mult_result[13]) begin
                result_exp = result_exp + 1;
                result_mant = mult_result[13:7];
            end else begin
                result_mant = mult_result[12:6];
            end
            
            // Check for overflow/underflow
            if (result_exp[7] && !result_exp[6]) begin
                // Underflow
                calculate_bf16_multiply = {result_sign, 8'h00, 7'h00};
            end else if (!result_exp[7] && result_exp[6]) begin
                // Overflow
                calculate_bf16_multiply = {result_sign, 8'hFF, 7'h00};
            end else begin
                calculate_bf16_multiply = {result_sign, result_exp, result_mant};
            end
        end
    endfunction
    
    // Monitor
    initial begin
        $monitor("Time=%0t rst_n=%b a=%h b=%h result=%h",
                 $time, rst_n, a, b, result);
    end

endmodule 