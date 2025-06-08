#!/bin/bash
# FP32 multiplier testbench simulation script for Unix-like systems

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the current directory
CURRENT_DIR="$(pwd)"

# Only change directory if we're not already in the correct directory
if [ "$CURRENT_DIR" != "$SCRIPT_DIR" ]; then
    echo "Changing to directory: $SCRIPT_DIR"
    cd "$SCRIPT_DIR"
fi

# Clean up old files
rm -f fp32_mul_sim dump.vcd sim.out fp32_ref_model_exec fp32_expected.txt fp32_refc_details.txt

# Compile C reference model
echo "Compiling C reference model..."
gcc -o fp32_ref_model_exec fp32_ref_model.c -lm
if [ $? -eq 0 ]; then
    echo "C reference model compiled successfully"
    ./fp32_ref_model_exec
else
    echo "Error compiling C reference model"
    exit 1
fi

# Compile and run the simulation
echo "Compiling and running simulation..."
iverilog -g2012 -o fp32_mul_sim fp32_mul_tb.sv ../mac_array/fp32_mul.v
if [ $? -eq 0 ]; then
    echo "Simulation compiled successfully"
    vvp fp32_mul_sim
else
    echo "Error compiling simulation"
    exit 1
fi
