# mask_resource_calculator.py

import argparse
import numpy as np
from sparse_transformer_mask import (
    create_normal_mask_step_by_step,
    create_strided_mask_step_by_step,
    create_fixed_mask_step_by_step
)

def calculate_worst_case_resources(mask, num_multiplications, mask_size):
    """
    Calculates the number of unique rows and columns needed for the first
    num_multiplications non-zero elements encountered in row-major order.
    This aims to find a 'worst-case' spread of indices for a given N.
    Also returns the list of these (row, col) points.
    """
    unique_rows = set()
    unique_cols = set()
    operations_count = 0
    worst_case_points = [] # New list to store points

    for i in range(mask_size):
        for j in range(mask_size):
            if mask[i, j] != 0: # Use numpy indexing
                unique_rows.add(i)
                unique_cols.add(j)
                worst_case_points.append((i, j)) # Add the point
                operations_count += 1
                if operations_count == num_multiplications:
                    return len(unique_rows), len(unique_cols), worst_case_points # Return points
    
    # If fewer non-zero elements than num_multiplications exist in the mask
    return len(unique_rows), len(unique_cols), worst_case_points # Return points

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate mask resources for sparse matrix multiplications.')
    parser.add_argument('--mask_size', type=int, default=1024, help='Size of the square mask matrix (default: 1024)')
    parser.add_argument('--num_multiplications', type=int, default=64, help='Total number of simultaneous multiplications (default: 64)')
    parser.add_argument('--window_size', type=int, default=32, help='Size of the local attention window for strided/fixed masks (default: 32)')
    parser.add_argument('--stride', type=int, default=32, help='Stride for strided attention (default: 32)')
    
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
    print(f"Calculating worst-case resources for {NUM_MULTIPLICATIONS} multiplications with MASK_SIZE={MASK_SIZE}:\n")

    # Normal Mask
    print("--- Normal Mask ---")
    normal_mask = create_normal_mask_step_by_step(MASK_SIZE)
    
    rows_normal, cols_normal, points_normal = calculate_worst_case_resources(normal_mask, NUM_MULTIPLICATIONS, MASK_SIZE)
    print(f"Required Unique Rows: {rows_normal}")
    print(f"Required Unique Columns: {cols_normal}")
    print(f"Total Unique Indices (Rows + Columns): {rows_normal + cols_normal}")
    print(f"Worst Case Points: {points_normal}\n")

    # Strided Mask
    print("--- Strided Mask ---")
    strided_mask = create_strided_mask_step_by_step(MASK_SIZE, WINDOW_SIZE, STRIDE)
    
    rows_strided, cols_strided, points_strided = calculate_worst_case_resources(strided_mask, NUM_MULTIPLICATIONS, MASK_SIZE)
    print(f"Required Unique Rows: {rows_strided}")
    print(f"Required Unique Columns: {cols_strided}")
    print(f"Total Unique Indices (Rows + Columns): {rows_strided + cols_strided}")
    print(f"Worst Case Points: {points_strided}\n")

    # Fixed Mask
    print("--- Fixed Mask ---")
    fixed_mask = create_fixed_mask_step_by_step(MASK_SIZE, WINDOW_SIZE)
    
    rows_fixed, cols_fixed, points_fixed = calculate_worst_case_resources(fixed_mask, NUM_MULTIPLICATIONS, MASK_SIZE)
    print(f"Required Unique Rows: {rows_fixed}")
    print(f"Required Unique Columns: {cols_fixed}")
    print(f"Total Unique Indices (Rows + Columns): {rows_fixed + cols_fixed}")
    print(f"Worst Case Points: {points_fixed}\n")
