# Create work library
vlib work

# Compile all files
vlog -work work ../mac_array/bf16_multiplier.v
vlog -work work bf16_multiplier_tb.v

# Create a file list for reference
vdir -lib work > file_list.txt 