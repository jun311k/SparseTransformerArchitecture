# Create work library
vlib work

# Compile all files
vlog -work work ../mac_array/bf16_multiplier.v
vlog -work work bf16_multiplier_tb.v

# Start simulation
vsim -novopt work.bf16_multiplier_tb

# Add waves
add wave -position insertpoint sim:/bf16_multiplier_tb/*
add wave -position insertpoint sim:/bf16_multiplier_tb/dut/*

# Run simulation
run -all

# Zoom to fit
wave zoom full 