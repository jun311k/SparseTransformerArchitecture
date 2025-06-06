#!/bin/bash
# FP32 multiplier testbench simulation script
cd npu_system/mac_array_sim && \
rm -f simv dump.vcd sim.out fp32_ref_model_exec expected_results.txt test_inputs.txt process_reason.txt && \
echo "Generating test inputs..." && \
cat << EOF > test_inputs.txt
3F800000 40000000
BF800000 40000000
3F000000 3F000000
40490FDB 40490FDB
3F800000 BF800000
00000000 3F800000
7F800000 3F800000
7FC00000 3F800000
7F800000 00000000
7F800000 FF800000
7F7FFFFF 7F7FFFFF
00800000 00800000
7F7FFFFF 00000001
00400000 3F800000
00400000 00400000
00000001 7F7FFFFF
EOF
echo "Compiling C code..." && \
gcc -o fp32_ref_model_exec fp32_ref_model.c -lm && \
echo "Generating expected results..." && \
./fp32_ref_model_exec && \
echo "Compiling Verilog/SystemVerilog code..." && \
iverilog -g2012 -s fp32_mul_tb -o simv fp32_mul_tb.sv ../mac_array/fp32_mul.v && \
echo "Running simulation..." && \
vvp simv
