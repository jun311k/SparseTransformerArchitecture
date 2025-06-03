#!/bin/bash
# BF16 multiplier testbench simulation script
iverilog -g2012 -o simv bf16_multiplier_tb.sv ../mac_array/bf16_multiplier.v && vvp simv 