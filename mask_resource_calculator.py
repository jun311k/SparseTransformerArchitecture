# mask_resource_calculator.py

import argparse
import numpy as np
import re
from sparse_transformer_mask import (
    create_normal_mask_step_by_step,
    create_strided_mask_step_by_step,
    create_fixed_mask_step_by_step
)

def print_cal_points_resources(mask, num_multiplications, mask_size, fname=None):
    """
    Calculates the number of unique rows and columns needed for the first
    num_multiplications non-zero elements encountered in row-major order.
    This aims to find a 'worst-case' spread of indices for a given N.
    Also returns the list of these (row, col) points.
    """
    cal_points = []
    unique_rows = set()
    unique_cols = set()
    operations_count = 0
    clks = 0
    fp = open(fname, 'a') if fname else None
    for i in range(mask_size):
        for j in range(mask_size):
            if mask[i, j] != 0: # Use numpy indexing
                cal_points.append((i, j)) # Store the point
                unique_rows.add(i)
                unique_cols.add(j)
                operations_count += 1
                if operations_count == num_multiplications:
                    # print cal_points one by one with comma separation in a line
                    outstr =(f"at time {clks}: {', '.join(f'({i}, {j})' for i, j in cal_points)}")
                    if fp:
                        fp.write(outstr + '\n')
                    else:
                        print(outstr)
                    clks += 1
                    cal_points.clear()
                    unique_rows.clear()
                    unique_cols.clear()
                    operations_count = 0
        
    # If fewer non-zero elements than num_multiplications exist in the mask
    if operations_count:
        outstr =(f"at time {clks}: {', '.join(f'({i}, {j})' for i, j in cal_points)}")
        if fp:
            fp.write(outstr + '\n')
        else:
            print(outstr)

def write_parameters_to_file(fname, MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, type = "normal"):
    with open(fname, 'w') as f:
        f.write(f"# Type: {type}\n")
        f.write(f"# Mask Size: {MASK_SIZE}\n")
        f.write(f"# Number of Multiplications: {NUM_MULTIPLICATIONS}\n")
        f.write(f"# Window Size: {WINDOW_SIZE}\n")
        f.write(f"# Stride: {STRIDE}\n")

def analyze_cal_points(fname):
    """
    Analyzes the cal points file to extract unique rows and columns.
    """
    rows = set()
    cols = set()
    max_row_and_col = 0
    max_case_rows = None
    max_case_cols = None
    max_case_points = None
    max_case_time = None
    cal_count = 0
    with open(fname, 'r') as f:
        for line in f:
            # get parameters from the # lines
            if line.startswith('#'):
                print(line.strip())
            # Check if the line contains 'at time' to find cal points
            elif line.startswith('at time'):
                cal_count += 1
                # parse the points
                # at time 0: (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)
                # to [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
                
                time_str = line.split(':')[0].strip()  # Extract the time part
                time_val = time_str.split(' ')[-1]  # Get the time value (e.g., '0' from 'at time 0')
                # remove 'at time 0: ' from the line
                points_string = line.split(': ')[1].strip()
                # Find all occurrences of (digit, digit)
                matches = re.findall(r'\((\d+),\s*(\d+)\)', points_string)
                # Convert matches to a list of integer tuples
                points_list = [(int(x), int(y)) for x, y in matches]
                # print(points_list)
                for point in points_list:
                    row, col = point
                    rows.add(row)
                    cols.add(col)
                # Update max row and col if this case has more unique indices
                if max_row_and_col < len(rows) + len(cols):
                    max_row_and_col = len(rows) + len(cols)
                    max_case_rows = sorted(rows)
                    max_case_cols = sorted(cols)
                    max_case_time = time_val
                    max_case_points = points_list
                    
                # Clear the sets for the next iteration
                rows.clear()
                cols.clear()

    print("\n--- Analysis Results ---")
    print(f"File: {fname}")
    print(f"Total Calculation Points: {cal_count}")
    print(f"Max Case Time: {max_case_time}")
    print(f"Unique Rows: {len(rows)}")
    print(f"Unique Columns: {len(cols)}")
    print(f"Total Unique Indices (Rows + Columns): {max_row_and_col}")
    # print(f"Max Case Points: {', '.join(f'({r}, {c})' for r, c in max_case_points)}")
    print(f"Max Case Row List: {max_case_rows}")
    print(f"Max Case Column List: {max_case_cols}")

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate mask resources for sparse matrix multiplications.')
    parser.add_argument('--mask_size', type=int, default=1024, help='Size of the square mask matrix (default: 1024)')
    parser.add_argument('--num_multiplications', type=int, default=64, help='Total number of simultaneous multiplications (default: 64)')
    parser.add_argument('--window_size', type=int, default=32, help='Size of the local attention window for strided/fixed masks (default: 32)')
    parser.add_argument('--stride', type=int, default=32, help='Stride for strided attention (default: 32)')
    parser.add_argument('--file', action='store_true', help='Enable file output for results')

    
    args = parser.parse_args()

    MASK_SIZE = args.mask_size
    NUM_MULTIPLICATIONS = args.num_multiplications
    WINDOW_SIZE = args.window_size
    STRIDE = args.stride

    print("--- Input Parameters ---")
    print(f"Mask Size: {MASK_SIZE}")
    print(f"Number of Multiplications: {NUM_MULTIPLICATIONS}")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}\n")
    print(f"file output enabled: {args.file}\n")

    print("*" * 50)
    print(f"Calculating worst-case resources for {NUM_MULTIPLICATIONS} multiplications with MASK_SIZE={MASK_SIZE}:\n")

    # Normal Mask
    print("--- Normal Mask ---")
    normal_mask = create_normal_mask_step_by_step(MASK_SIZE)

    # print parameters in the outputfile with '#' as comment
    fname = f"generated/normal_mask_{NUM_MULTIPLICATIONS}.txt" if args.file else None
    if fname:
        write_parameters_to_file(fname, MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, type="normal")
    
    print_cal_points_resources(normal_mask, NUM_MULTIPLICATIONS, MASK_SIZE, fname = fname)
    if fname:
        analyze_cal_points(fname)

    # do the same for strided and fixed masks
    print("\n" + "*" * 50)
    print("--- Strided Mask ---")
    strided_mask = create_strided_mask_step_by_step(MASK_SIZE, WINDOW_SIZE, STRIDE)
    fname = f"generated/strided_mask_{NUM_MULTIPLICATIONS}.txt" if args.file else None
    if fname:
        write_parameters_to_file(fname, MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, type="strided")
    print_cal_points_resources(strided_mask, NUM_MULTIPLICATIONS, MASK_SIZE, fname = fname)
    if fname:
        analyze_cal_points(fname)

    print("\n" + "*" * 50)
    print("--- Fixed Mask ---")
    fixed_mask = create_fixed_mask_step_by_step(MASK_SIZE, WINDOW_SIZE)
    fname = f"generated/fixed_mask_{NUM_MULTIPLICATIONS}.txt" if args.file else None
    if fname:
        write_parameters_to_file(fname, MASK_SIZE, NUM_MULTIPLICATIONS, WINDOW_SIZE, STRIDE, type="fixed")

    print_cal_points_resources(fixed_mask, NUM_MULTIPLICATIONS, MASK_SIZE, fname = fname)
    if fname:
        analyze_cal_points(fname)

    print("\n" + "*" * 50)
    print("Resource calculation completed.")
    print("Check the generated files in the 'generated' directory for detailed results.")
