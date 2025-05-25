"""
Sparse Transformer Attention Mask Implementation

This module implements sparse attention patterns for transformer models,
specifically designed for CIFAR-10 image data (32x32 pixels = 1024 tokens).
It provides functions to create, visualize, and analyze two types of sparse attention patterns:
1. Strided pattern
2. Fixed pattern

These patterns significantly reduce computational complexity while maintaining
the model's ability to capture both local and global dependencies.

References:
- Sparse Transformer (Child et al., 2019): https://arxiv.org/abs/1904.10509
- Longformer (Beltagy et al., 2020): https://arxiv.org/abs/2004.05150
- BigBird (Zaheer et al., 2020): https://arxiv.org/abs/2007.14062
"""

import numpy as np
import matplotlib.pyplot as plt

# Custom implementation of mask functions
def create_zero_mask(size):
    """
    Create a zero mask of the specified size.
    
    Args:
        size (int): Size of the square mask matrix
        
    Returns:
        numpy.ndarray: A square matrix of zeros with shape (size, size)
    """
    return np.zeros((size, size), dtype=np.float32)

def add_diagonal(mask, value=1):
    """
    Add diagonal values to the mask.
    
    Args:
        mask (numpy.ndarray): Input mask matrix
        value (int, optional): Value to set on the diagonal. Defaults to 1.
        
    Returns:
        numpy.ndarray: Mask with diagonal values set
    """
    mask_copy = mask.copy()
    np.fill_diagonal(mask_copy, value)
    return mask_copy

def add_local_window(mask, window_size=32, pattern_type="strided", value=2, overwrite=False):
    """
    Add local window attention to the mask.
    
    Args:
        mask (numpy.ndarray): Input mask matrix
        window_size (int, optional): Size of the local attention window. Defaults to 32.
        pattern_type (str, optional): Type of pattern - "strided" or "fixed". Defaults to "strided".
        value (int, optional): Value to set for local window attention. Defaults to 2.
        overwrite (bool, optional): Whether to overwrite existing values. Defaults to False.
        
    Returns:
        numpy.ndarray: Mask with local window attention added
    """
    mask_copy = mask.copy()
    size = mask_copy.shape[0]
    
    if pattern_type == "strided":
        # For strided pattern: local window is a band below the diagonal
        for i in range(size):
            # Start index is max(0, i-(window_size-1))
            start_idx = max(0, i - (window_size - 1))
            # End index is diagonal point minus 1
            if overwrite:
                end_idx = i
            else:
                end_idx = i-1
            # Fill the local window
            for j in range(start_idx, end_idx+1):
                mask_copy[i, j] = value
    
    elif pattern_type == "fixed":
        # For fixed pattern: local window is block-diagonal
        for i in range(size):
            # Calculate the block this position belongs to
            block_idx = i // window_size
            # Start and end of the current block
            block_start = block_idx * window_size
            block_end = min(size, (block_idx + 1) * window_size)
            
            # Fill local block (excluding diagonal)
            for j in range(block_start, min(block_end, i)):
                mask_copy[i, j] = value
    
    return mask_copy

def add_strided_attention(mask, window_size=32, stride=32, value=3):
    """
    Add strided attention to the mask.
    
    Args:
        mask (numpy.ndarray): Input mask matrix
        window_size (int, optional): Size of the local attention window. Defaults to 32.
        stride (int, optional): Stride between attention points. Defaults to 32.
        value (int, optional): Value to set for strided attention. Defaults to 3.
        
    Returns:
        numpy.ndarray: Mask with strided attention added
    """
    mask_copy = mask.copy()
    size = mask_copy.shape[0]
    
    # For each query position
    for i in range(size):
        # Only apply strided attention beyond the local window
        if i >= window_size:
            # First strided position is just below the local window
            first_strided_pos = i - window_size
            
            # Set first strided position
            if first_strided_pos >= 0:
                mask_copy[i, first_strided_pos] = value
            
            # Set remaining strided positions
            pos = first_strided_pos - stride
            while pos >= 0:
                mask_copy[i, pos] = value
                pos -= stride
    
    return mask_copy

def add_column_attention(mask, window_size=32, value=3):
    """
    Add column attention for fixed pattern.
    
    Args:
        mask (numpy.ndarray): Input mask matrix
        window_size (int, optional): Size of each block. Defaults to 32.
        value (int, optional): Value to set for column attention. Defaults to 3.
        
    Returns:
        numpy.ndarray: Mask with column attention added
    """
    mask_copy = mask.copy()
    size = mask_copy.shape[0]
    
    # For each query position
    for i in range(size):
        # Calculate the block this position belongs to
        block_idx = i // window_size
        
        # Set attention to the last element of each previous block
        for b in range(block_idx):
            # Last element of each previous block
            last_element_idx = min(size - 1, (b + 1) * window_size - 1)
            mask_copy[i, last_element_idx] = value
    
    return mask_copy

def create_strided_mask_step_by_step(size, window_size=32, stride=32):
    """
    Create a strided mask step by step.
    
    Args:
        size (int): Size of the square mask matrix
        window_size (int, optional): Size of the local attention window. Defaults to 32.
        stride (int, optional): Stride between attention points. Defaults to 32.
        
    Returns:
        numpy.ndarray: Complete strided attention mask
    """
    # Step 1: Create zero mask
    mask = create_zero_mask(size)
    
    # Step 2: Add diagonal values
    mask = add_diagonal(mask, value=1)
    
    # Step 3: Add local window attention
    mask = add_local_window(mask, window_size, pattern_type="strided", value=2)
    
    # Step 4: Add strided attention
    mask = add_strided_attention(mask, window_size, stride, value=3)
    
    return mask

def create_fixed_mask_step_by_step(size, window_size=32):
    """
    Create a fixed pattern mask step by step.
    
    Args:
        size (int): Size of the square mask matrix
        window_size (int, optional): Size of each block. Defaults to 32.
        
    Returns:
        numpy.ndarray: Complete fixed pattern attention mask
    """
    # Step 1: Create zero mask
    mask = create_zero_mask(size)
    
    # Step 2: Add diagonal values
    mask = add_diagonal(mask, value=1)
    
    # Step 3: Add local window attention with fixed pattern
    mask = add_local_window(mask, window_size, pattern_type="fixed", value=2)
    
    # Step 4: Add column attention
    mask = add_column_attention(mask, window_size, value=3)
    
    return mask

def visualize_mask_sample(mask, title, sample_size, colormap, save_path):
    """
    Visualize a sample of the mask and save it.
    
    Args:
        mask (numpy.ndarray): Input mask matrix
        title (str): Title for the visualization
        sample_size (int): Size of the sample to visualize
        colormap: Matplotlib colormap to use
        save_path (str): Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(mask[0:sample_size, 0:sample_size], cmap=colormap, vmin=0, vmax=3)
    plt.title(f'{title} ({sample_size}x{sample_size} sample)')
    plt.grid(True)
    
    # Add colorbar
    cbar = plt.colorbar(ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['No Attention', 'Diagonal', 'Local', 'Strided/Column'])
    
    # Calculate sparsity
    sparsity = 1.0 - np.count_nonzero(mask) / mask.size
    sample_sparsity = 1.0 - np.count_nonzero(mask[0:sample_size, 0:sample_size]) / (sample_size * sample_size)
    
    # Add sparsity annotation
    plt.annotate(f'Full Mask Sparsity: {sparsity:.4%}\nSample Sparsity: {sample_sparsity:.4%}', 
                xy=(0.02, 0.02), 
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save the visualization
    plt.savefig(save_path)
    print(f"Mask visualization saved as {save_path}")
    
    # Show the plot
    plt.show()

def convert_to_binary_mask(mask):
    """
    Convert a mask to a binary mask.
    
    Args:
        mask (numpy.ndarray): Input mask with values 0, 1, 2, 3
        
    Returns:
        numpy.ndarray: Binary mask with values 0 and 1
    """
    return np.where(mask > 0, 1, 0)

def add_lower_triangle(mask, value=2):
    """
    Add lower triangular values to the mask (excluding diagonal).
    
    Args:
        mask (numpy.ndarray): Input mask matrix
        value (int, optional): Value to set for lower triangular part. Defaults to 3.
        
    Returns:
        numpy.ndarray: Mask with lower triangular values set
    """
    mask_copy = mask.copy()
    size = mask_copy.shape[0]
    
    # For each row
    for i in range(size):
        # Fill all columns before the diagonal
        for j in range(i):
            mask_copy[i, j] = value
    
    return mask_copy

def create_normal_mask_step_by_step(size):
    """
    Create a normal (full) attention mask step by step.
    
    Args:
        size (int): Size of the square mask matrix
        
    Returns:
        numpy.ndarray: Complete normal attention mask
    """
    # Step 1: Create zero mask
    mask = create_zero_mask(size)
    
    # Step 2: Add diagonal values
    mask = add_diagonal(mask, value=1)
    
    # Step 3: Add lower triangular values
    mask = add_lower_triangle(mask, value=2)
    
    return mask

def visualize_mask_comparison(masks, titles, sample_size, colormap, save_path):
    """
    Visualize multiple masks side by side for comparison.
    
    Args:
        masks (list): List of mask matrices
        titles (list): List of titles for each mask
        sample_size (int): Size of the sample to visualize
        colormap: Matplotlib colormap to use
        save_path (str): Path to save the visualization
    """
    n_masks = len(masks)
    fig, axes = plt.subplots(1, n_masks, figsize=(6*n_masks, 8))
    
    for i, (mask, title, ax) in enumerate(zip(masks, titles, axes)):
        im = ax.imshow(mask[0:sample_size, 0:sample_size], cmap=colormap, vmin=0, vmax=3)
        ax.set_title(f'{title}\n({sample_size}x{sample_size} sample)')
        ax.grid(True)
        
        # Calculate sparsity
        sparsity = 1.0 - np.count_nonzero(mask) / mask.size
        sample_sparsity = 1.0 - np.count_nonzero(mask[0:sample_size, 0:sample_size]) / (sample_size * sample_size)
        
        # Add sparsity annotation
        ax.annotate(f'Sparsity: {sparsity:.4%}', 
                    xy=(0.02, 0.02), 
                    xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['No Attention', 'Diagonal', 'Local', 'Strided/Column'])
    
    # Save the visualization
    plt.savefig(save_path)
    print(f"Comparison visualization saved as {save_path}")
    
    # Show the plot
    plt.show()

def add_sliding_window(mask, window_size=32, value=2):
    """
    Add sliding window attention pattern.
    
    Args:
        mask (numpy.ndarray): Input mask matrix
        window_size (int, optional): Size of the sliding window. Defaults to 32.
        value (int, optional): Value to set for sliding window attention. Defaults to 2.
        
    Returns:
        numpy.ndarray: Mask with sliding window attention added
    """
    mask_copy = mask.copy()
    size = mask_copy.shape[0]
    
    # For each query position
    for i in range(size):
        # Calculate window boundaries
        start_idx = max(0, i - window_size // 2)
        end_idx = min(size, i + window_size // 2 + 1)
        
        # Fill the sliding window (excluding diagonal)
        for j in range(start_idx, end_idx):
            if j != i:  # Skip diagonal
                mask_copy[i, j] = value
    
    return mask_copy

def add_dilated_sliding_window(mask, window_size=32, dilation=2, value=2):
    """
    Add dilated sliding window attention pattern.
    각 row의 중심(자기 자신)을 기준으로 window_size만큼 dilation 간격으로 좌우에 점이 찍히는 형태.
    
    Args:
        mask (numpy.ndarray): Input mask matrix
        window_size (int, optional): Number of attended positions (must be odd). Defaults to 32.
        dilation (int, optional): Dilation factor. Defaults to 2.
        value (int, optional): Value to set for dilated sliding window attention. Defaults to 2.
        
    Returns:
        numpy.ndarray: Mask with dilated sliding window attention added
    """
    mask_copy = mask.copy()
    size = mask_copy.shape[0]
    half = window_size // 2
    
    for i in range(size):
        for k in range(-half, half+1):
            j = i + k * dilation
            if 0 <= j < size and j != i:
                mask_copy[i, j] = value
    return mask_copy

def create_sliding_window_mask_step_by_step(size, window_size=32):
    """
    Create a sliding window attention mask step by step.
    
    Args:
        size (int): Size of the square mask matrix
        window_size (int, optional): Size of the sliding window. Defaults to 32.
        
    Returns:
        numpy.ndarray: Complete sliding window attention mask
    """
    # Step 1: Create zero mask
    mask = create_zero_mask(size)
    
    # Step 2: Add diagonal values
    mask = add_diagonal(mask, value=1)
    
    # Step 3: Add sliding window attention
    mask = add_sliding_window(mask, window_size, value=2)
    
    return mask

def create_dilated_sliding_window_mask_step_by_step(size, window_size=32, dilation=2):
    """
    Create a dilated sliding window attention mask step by step.
    
    Args:
        size (int): Size of the square mask matrix
        window_size (int, optional): Size of the sliding window. Defaults to 32.
        dilation (int, optional): Dilation factor. Defaults to 2.
        
    Returns:
        numpy.ndarray: Complete dilated sliding window attention mask
    """
    # Step 1: Create zero mask
    mask = create_zero_mask(size)
    
    # Step 2: Add diagonal values
    mask = add_diagonal(mask, value=1)
    
    # Step 3: Add dilated sliding window attention
    mask = add_dilated_sliding_window(mask, window_size, dilation, value=2)
    
    return mask

# Main function
if __name__ == "__main__":
    # Parameters
    size = 1024
    window_size = 32
    stride = 32
    custom_cmap = plt.cm.colors.ListedColormap(['lightgray', 'darkblue', 'royalblue', 'skyblue'])
    
    # Create normal attention mask
    print("Creating Normal attention mask step by step...")
    normal_mask = create_normal_mask_step_by_step(size)
    print("Normal attention mask created!")
    
    # Calculate sparsity
    normal_sparsity = 1.0 - np.count_nonzero(normal_mask) / normal_mask.size
    print(f"Normal attention sparsity: {normal_sparsity:.4%}")
    
    # Visualize normal mask
    visualize_mask_sample(
        mask=normal_mask,
        title='Normal (Full) Attention Mask',
        sample_size=128,
        colormap=custom_cmap,
        save_path='normal_mask_128x128.png'
    )

    print("Creating Strided mask step by step...")
    
    # Create strided mask using the integrated function
    strided_mask = create_strided_mask_step_by_step(size, window_size, stride)
    print("Strided mask created!")
    
    # Calculate sparsity
    sparsity = 1.0 - np.count_nonzero(strided_mask) / strided_mask.size
    print(f"Sparsity: {sparsity:.4%}")
    
    # Visualize 64x64 sample of strided mask
    visualize_mask_sample(
        mask=strided_mask,
        title='Strided Pattern Mask',
        sample_size=64,
        colormap=custom_cmap,
        save_path='strided_mask_64x64.png'
    )
    
    # Visualize 128x128 sample of strided mask
    visualize_mask_sample(
        mask=strided_mask,
        title='Strided Pattern Mask',
        sample_size=128,
        colormap=custom_cmap,
        save_path='strided_mask_128x128.png'
    )

    # Now create and visualize the fixed pattern mask
    print("\nCreating Fixed pattern mask step by step...")
    
    # Create fixed pattern mask using the integrated function
    fixed_mask = create_fixed_mask_step_by_step(size, window_size)
    print("Fixed pattern mask created!")
    
    # Calculate sparsity
    fixed_sparsity = 1.0 - np.count_nonzero(fixed_mask) / fixed_mask.size
    print(f"Fixed pattern sparsity: {fixed_sparsity:.4%}")
    
    # Visualize 128x128 sample of fixed mask
    visualize_mask_sample(
        mask=fixed_mask,
        title='Fixed Pattern Mask',
        sample_size=128,
        colormap=custom_cmap,
        save_path='fixed_mask_128x128.png'
    )
    
    # Create comparison visualization of all three mask types
    print("\nCreating comparison visualization...")
    visualize_mask_comparison(
        masks=[normal_mask[:128, :128], strided_mask[:128, :128], fixed_mask[:128, :128]],
        titles=['Normal (Full) Attention', 'Strided Pattern', 'Fixed Pattern'],
        sample_size=128,
        colormap=custom_cmap,
        save_path='mask_comparison_128x128.png'
    )
    print("Comparison visualization created!")

    # Create sliding window mask
    print("\nCreating Sliding Window mask step by step...")
    sliding_window_mask = create_sliding_window_mask_step_by_step(
        size=size,
        window_size=window_size
    )
    print("Sliding Window mask created!")
    
    # Calculate sparsity
    sliding_window_sparsity = 1.0 - np.count_nonzero(sliding_window_mask) / sliding_window_mask.size
    print(f"Sliding Window attention sparsity: {sliding_window_sparsity:.4%}")
    
    # Visualize sliding window mask
    visualize_mask_sample(
        mask=sliding_window_mask,
        title='Sliding Window Pattern Mask',
        sample_size=128,
        colormap=custom_cmap,
        save_path='sliding_window_mask_128x128.png'
    )
    
    # Create dilated sliding window mask
    print("\nCreating Dilated Sliding Window mask step by step...")
    dilated_sliding_window_mask = create_dilated_sliding_window_mask_step_by_step(
        size=size,
        window_size=window_size,
        dilation=2
    )
    print("Dilated Sliding Window mask created!")
    
    # Calculate sparsity
    dilated_sparsity = 1.0 - np.count_nonzero(dilated_sliding_window_mask) / dilated_sliding_window_mask.size
    print(f"Dilated Sliding Window attention sparsity: {dilated_sparsity:.4%}")
    
    # Visualize dilated sliding window mask
    visualize_mask_sample(
        mask=dilated_sliding_window_mask,
        title='Dilated Sliding Window Pattern Mask',
        sample_size=128,
        colormap=custom_cmap,
        save_path='dilated_sliding_window_mask_128x128.png'
    )
    
    # Create comparison visualization including all patterns
    print("\nCreating comparison visualization...")
    visualize_mask_comparison(
        masks=[
            normal_mask[:128, :128],
            strided_mask[:128, :128],
            fixed_mask[:128, :128],
            sliding_window_mask[:128, :128],
            dilated_sliding_window_mask[:128, :128]
        ],
        titles=[
            'Normal (Full) Attention',
            'Strided Pattern',
            'Fixed Pattern',
            'Sliding Window',
            'Dilated Sliding Window'
        ],
        sample_size=128,
        colormap=custom_cmap,
        save_path='mask_comparison_128x128.png'
    )
    print("Comparison visualization created!") 