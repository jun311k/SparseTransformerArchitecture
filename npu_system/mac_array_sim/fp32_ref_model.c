#include <stdio.h>
#include <stdint.h>
#include <string.h> // For memcpy
#include <math.h>   // For isnan, isinf
#include <stdlib.h> // For strtoul

// Helper to convert uint32_t to float using memcpy (safer for strict aliasing)
static float u32_to_float_safe(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

// Helper to convert float to uint32_t using memcpy
static uint32_t float_to_u32_safe(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(uint32_t));
    return u;
}

// Function to check if a float (represented as u32) is subnormal
// A subnormal number has exponent field 0 and mantissa non-zero.
static int is_subnormal_from_bits(uint32_t val_bits) {
    uint32_t exp_val = (val_bits >> 23) & 0xFF;
    uint32_t man_val = val_bits & 0x007FFFFF;
    return (exp_val == 0 && man_val != 0);
}

uint32_t calculate_fp32_c_native_float(uint32_t a_bits, uint32_t b_bits) {
    // --- Pre-computation: Handle special inputs to match DUT behavior if necessary ---
    uint32_t sign_a = (a_bits >> 31) & 0x01;
    uint32_t exp_a  = (a_bits >> 23) & 0xFF;
    uint32_t man_a  = a_bits & 0x007FFFFF;

    uint32_t sign_b = (b_bits >> 31) & 0x01;
    uint32_t exp_b  = (b_bits >> 23) & 0xFF;
    uint32_t man_b  = b_bits & 0x007FFFFF;

    // 1. NaN propagation (ensure specific qNaN if DUT produces one)
    if ((exp_a == 0xFF && man_a != 0) || (exp_b == 0xFF && man_b != 0)) {
        return 0x7FC00000; // qNaN (example, match DUT's qNaN)
    }

    // 2. Infinity * Zero = NaN (IEEE 754 rule)
    int a_is_inf = (exp_a == 0xFF && man_a == 0);
    int b_is_inf = (exp_b == 0xFF && man_b == 0);
    // Use strict zero (exponent and mantissa are zero) for this rule.
    // If DUT flushes subnormal inputs to zero, this check might need adjustment
    // or inputs should be flushed before this check.
    int a_is_strict_zero = (exp_a == 0 && man_a == 0);
    int b_is_strict_zero = (exp_b == 0 && man_b == 0);

    if ((a_is_inf && b_is_strict_zero) || (b_is_inf && a_is_strict_zero)) {
        return 0x7FC00000; // NaN
    }

    // Convert to float
    float val_a = u32_to_float_safe(a_bits);
    float val_b = u32_to_float_safe(b_bits);

    // --- Optional: FTZ for inputs if DUT does this ---
    // This example assumes DUT does NOT FTZ inputs if it supports subnormals,
    // or that subnormals are handled by the C float type directly.
    // If DUT FTZ inputs:
    // if (exp_a == 0 && man_a != 0) val_a = copysignf(0.0f, val_a);
    // if (exp_b == 0 && man_b != 0) val_b = copysignf(0.0f, val_b);

    // --- Perform multiplication ---
    // Note: C's default rounding is typically round-to-nearest-ties-to-even.
    // If DUT uses a different rounding, <fenv.h> might be an option, or this approach is unsuitable.
    float result_val = val_a * val_b;
    uint32_t result_bits = float_to_u32_safe(result_val);

    // --- Post-computation: Adjust result to match DUT behavior if necessary ---

    // 1. Ensure specific NaN pattern for results that become NaN
    if (isnan(result_val)) {
        // This catches NaNs produced by the multiplication itself (e.g., 0 * Inf if C handles it that way)
        // The earlier checks for input NaN and Inf*0 should already be handled.
        return 0x7FC00000; // qNaN
    }

    // 2. FTZ for subnormal results if DUT requires it
    // (Standard C float result might be subnormal here)
    // If DUT flushes subnormal results to zero:
    // uint32_t res_exp = (result_bits >> 23) & 0xFF;
    // uint32_t res_man = result_bits & 0x007FFFFF;
    // if (res_exp == 0 && res_man != 0) { // If result is subnormal
    //     uint32_t res_sign = (result_bits >> 31) & 0x01;
    //     return (res_sign << 31); // Return signed zero
    // }
    // The above FTZ check is an example. The exact condition for FTZ
    // (e.g. based on exponent value before rounding) might be more complex
    // and is one reason manual bitwise models are used.

    return result_bits;
}

int main(int argc, char *argv[]) {
    FILE *input_file, *output_file, *fp_reason;
    uint32_t a_bits, b_bits, result;
    char line[256];
    char hex_a[9], hex_b[9]; // 8 hex chars + null terminator
    int line_count = 0;
    int test_index = 0;  // 0-based test index

    input_file = fopen("fp32_inputs.txt", "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error opening fp32_inputs.txt for reading.\n");
        return 1;
    }

    output_file = fopen("fp32_expected.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error opening fp32_expected.txt for writing.\n");
        fclose(input_file);
        return 1;
    }

    fp_reason = fopen("fp32_refc_details.txt", "w");
    if (fp_reason == NULL) {
        fprintf(stderr, "Error opening fp32_refc_details.txt for writing.\n");
        fclose(input_file);
        fclose(output_file);
        return 1;
    }

    // Read line by line, skip comments and empty lines
    while (fgets(line, sizeof(line), input_file)) {
        line_count++;
        
        // Skip empty lines and comments
        if (line[0] == '\n' || line[0] == '/' || line[0] == ' ' || line[0] == '\r') {
            continue;
        }

        // Try to read category and two hex values from the line
        int category;
        if (sscanf(line, "%d %8s %8s", &category, hex_a, hex_b) == 3) {
            a_bits = (uint32_t)strtoul(hex_a, NULL, 16);
            b_bits = (uint32_t)strtoul(hex_b, NULL, 16);
            result = calculate_fp32_c_native_float(a_bits, b_bits);
            fprintf(output_file, "%08X\n", result);
            // write them in floating point format and bits format
            fprintf(fp_reason, "Line %d (Index %d): Category=%d Input A: %s (%f), Input B: %s (%f), Expected_Result: %08X (%f)\n",
                    line_count, test_index, category, hex_a, u32_to_float_safe(a_bits),
                    hex_b, u32_to_float_safe(b_bits),
                    result, u32_to_float_safe(result));
            test_index++;  // Increment test index for valid test cases
        } else {
            fprintf(stderr, "Warning: Invalid format at line %d: %s", line_count, line);
        }
    }

    fclose(input_file);
    fclose(output_file);
    fclose(fp_reason);
    printf("Processing complete. Results written to fp32_expected.txt and fp32_refc_details.txt.\n");
    return 0;
}
