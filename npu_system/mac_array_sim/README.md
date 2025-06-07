# FP32 Multiplier Simulation

This directory contains the simulation environment for the FP32 multiplier implementation.

## Directory Structure

```
mac_array_sim/
├── fp32_mul_tb.sv          # Main testbench
├── fp32_ref_model.c        # C reference model
├── test_fp32_inputs.txt    # Test cases
├── test_fp32_expected.txt  # Expected results
├── run_sim_fp32_mul.ps1    # PowerShell script for Windows
├── run_sim_fp32_mul.bat    # Batch script for Windows
└── run_sim_fp32_mul.sh     # Shell script for Unix-like systems
```

## Test Cases

The test cases cover various scenarios:
- Normal operations without zero
- Normal operations with zero
- Special cases (Infinity, NaN)
- Edge cases
- Subnormal numbers

Each test case includes:
- Input A (32-bit FP32)
- Input B (32-bit FP32)
- Expected result (32-bit FP32)

## Running the Simulation

### Windows (PowerShell)
```powershell
.\run_sim_fp32_mul.ps1
```

### Windows (Command Prompt)
```cmd
run_sim_fp32_mul.bat
```

### Unix-like Systems
```bash
./run_sim_fp32_mul.sh
```

## Simulation Process

1. The script first compiles and runs the C reference model to generate expected results
2. Then it compiles the SystemVerilog testbench and DUT using iverilog
3. Finally, it runs the simulation using vvp

## Output Files

- `dump.vcd`: Waveform file for debugging
- `test_fp32_expected.txt`: Expected results from C reference model
- `test_fp32_refc_details.txt`: Detailed output from C reference model

## Requirements

- iverilog (for SystemVerilog simulation)
- gcc (for C reference model)
- Python 3.x (for test case generation)

## Notes

- The testbench uses SystemVerilog features for better test case management
- The C reference model implements IEEE 754 floating-point multiplication
- Test results are displayed with both hexadecimal and decimal values
- Special values (-0, +0, -Inf, +Inf, -NaN, +NaN) are properly handled and displayed 