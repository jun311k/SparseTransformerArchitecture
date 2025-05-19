# Sparse Transformer Attention Masks for CIFAR-10

This project implements and visualizes sparse attention patterns for CIFAR-10 image data (32x32 pixels = 1024 tokens). Sparse attention patterns significantly reduce computational complexity and memory usage while maintaining the model's ability to capture local and global dependencies.

## Attention Patterns

Two types of sparse attention patterns are implemented:

### 1. Strided Pattern

The strided pattern combines three types of attention:
- **Diagonal self-attention (value 1)**: Each token attends to itself
- **Local window attention (value 2)**: Each token attends to nearby tokens within a local window
- **Strided attention (value 3)**: Each token attends to previous tokens at fixed intervals

This pattern achieves approximately **95.4% sparsity** with a window size and stride of 32.

### 2. Fixed Pattern

The fixed pattern uses:
- **Diagonal self-attention (value 1)**: Each token attends to itself
- **Block-wise local attention (value 2)**: Each token attends to previous tokens within the same block
- **Fixed column attention (value 3)**: Each token attends to the last token of each previous block

This pattern achieves approximately **96.9% sparsity** with a block size of 32.

## Implementation Details

This project is implemented as a single Python file that provides:

- Mask generation functions (step-by-step implementation)
- Mask visualization tools
- Sparsity calculation and statistics

## Usage

### Generating Attention Masks

```python
from sparse_transformer_mask import create_strided_mask_step_by_step, create_fixed_mask_step_by_step

# Generate strided mask
strided_mask = create_strided_mask_step_by_step(size=1024, window_size=32, stride=32)

# Generate fixed mask
fixed_mask = create_fixed_mask_step_by_step(size=1024, window_size=32)
```

### Converting to Binary Masks

The generated masks have values 1, 2, and 3 to distinguish attention types, but for use in actual transformer models, they should be converted to binary masks (consisting only of 0s and 1s):

```python
import numpy as np
from sparse_transformer_mask import convert_to_binary_mask

# Convert to binary mask (all non-zero values to 1)
binary_strided_mask = convert_to_binary_mask(strided_mask)
binary_fixed_mask = convert_to_binary_mask(fixed_mask)

# Check binary mask sparsity
binary_sparsity = 1.0 - np.count_nonzero(binary_strided_mask) / binary_strided_mask.size
print(f"Binary mask sparsity: {binary_sparsity:.4%}")
```

### Visualizing Masks

```python
from sparse_transformer_mask import visualize_mask_sample
import matplotlib.pyplot as plt

# Mask visualization settings
custom_cmap = plt.cm.colors.ListedColormap(['lightgray', 'darkblue', 'royalblue', 'skyblue'])

# Visualize 64x64 sample
visualize_mask_sample(
    mask=strided_mask,
    title='Strided Pattern Mask',
    sample_size=64,
    colormap=custom_cmap,
    save_path='strided_mask_64x64.png'
)
```

### Running the Script Directly

```bash
python sparse_transformer_mask.py
```

This command generates Strided and Fixed pattern masks and visualizes 64x64 and 128x128 sized samples for each.

## Theoretical Efficiency

For a sequence length of 1024 (32x32 CIFAR-10 image):

- **Standard attention**: O(n²) = 1,048,576 operations
- **Sparse attention** (95% sparsity): O((1-s)·n²) = 52,428 operations
- **Theoretical speedup**: ~20x

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## References

- [Sparse Transformer (Child et al., 2019)](https://arxiv.org/abs/1904.10509)
- [Longformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
- [BigBird (Zaheer et al., 2020)](https://arxiv.org/abs/2007.14062) 