# Sparse Transformer Attention Masks for CIFAR-10

**한국어 버전은 [README_kr.md](README_kr.md)를 참고하세요.**

This project implements and visualizes sparse attention patterns for CIFAR-10 image data (32x32 pixels = 1024 tokens). Sparse attention patterns significantly reduce computational complexity and memory usage while maintaining the model's ability to capture local and global dependencies.

## Scripts Overview

This project contains two main scripts:

1. `sparse_transformer_mask.py`: Implements and visualizes different attention patterns
2. `mask_resource_calculator.py`: Analyzes resource requirements for mask operations

### 1. Sparse Transformer Mask Generator

The `sparse_transformer_mask.py` script focuses on generating and visualizing different attention patterns.

```bash
python sparse_transformer_mask.py --help
usage: sparse_transformer_mask.py [-h] [--size SIZE] [--window_size WINDOW_SIZE] [--stride STRIDE] 
                                 [--sample_size SAMPLE_SIZE] [--full_png] [--no_graphic] 
                                 [--order {row_first,column_first}] [--skip_pattern_print]

Sparse Transformer Attention Mask Implementation

options:
  -h, --help            show this help message and exit
  --size SIZE           Size of the square mask matrix (default: 1024)
  --window_size WINDOW_SIZE
                        Size of the local attention window (default: 32)
  --stride STRIDE       Stride between attention points (default: 32)
  --sample_size SAMPLE_SIZE
                        Size of the sample to visualize (default: 128)
  --full_png            Save full-size PNG images instead of samples
  --no_graphic          Bypass graphics and only print sparsity patterns
  --order {row_first,column_first}
                        Order of non-zero elements in the mask (default: row_first)
  --skip_pattern_print  Skip printing non-zero elements of the mask
```

### 2. Mask Resource Calculator

The `mask_resource_calculator.py` script analyzes the resource requirements for mask operations, including:
- Number of unique rows and columns needed
- Changes in required resources between consecutive operations
- Maximum resource usage cases

```bash
python mask_resource_calculator.py --help
usage: mask_resource_calculator.py [-h] [--mask_size MASK_SIZE] [--num_multiplications NUM_MULTIPLICATIONS]
                                   [--window_size WINDOW_SIZE] [--stride STRIDE] [--file] [--read_limit READ_LIMIT]

Calculate mask resources for sparse matrix multiplications.

options:
  -h, --help            show this help message and exit
  --mask_size MASK_SIZE
                        Size of the square mask matrix (default: 1024)
  --num_multiplications NUM_MULTIPLICATIONS
                        Total number of simultaneous multiplications (default: 64)
  --window_size WINDOW_SIZE
                        Size of the local attention window for strided/fixed masks (default: 32)
  --stride STRIDE       Stride for strided attention (default: 32)
  --file                Enable file output for results
  --read_limit READ_LIMIT
                        Limit the number of lines read from the file (default: 1000)
```

## Attention Patterns

Five types of attention patterns are implemented:

### 1. Normal (Full) Attention

The normal attention pattern implements standard causal attention:
- **Diagonal self-attention (value 1)**: Each token attends to itself
- **Lower triangular attention (value 2)**: Each token attends to all previous tokens

This pattern has **49.6% sparsity** (only upper triangular part is masked out).

![Normal Attention](images/normal_mask_128x128.png)

### 2. Strided Pattern

The strided pattern combines three types of attention:
- **Diagonal self-attention (value 1)**: Each token attends to itself
- **Local window attention (value 2)**: Each token attends to nearby tokens within a local window
- **Strided attention (value 3)**: Each token attends to previous tokens at fixed intervals

This pattern achieves approximately **95.4% sparsity** with a window size and stride of 32.

![Strided Pattern](images/strided_mask_128x128.png)

### 3. Fixed Pattern

The fixed pattern uses:
- **Diagonal self-attention (value 1)**: Each token attends to itself
- **Block-wise local attention (value 2)**: Each token attends to previous tokens within the same block
- **Fixed column attention (value 3)**: Each token attends to the last token of each previous block

This pattern achieves approximately **96.9% sparsity** with a block size of 32.

![Fixed Pattern](images/fixed_mask_128x128.png)

### 4. Sliding Window Pattern

Each token attends to a fixed-size window of neighboring tokens (excluding itself):
- **Diagonal self-attention (value 1)**
- **Sliding window attention (value 2)**

This pattern achieves approximately **96.8% sparsity** with a window size of 32.

![Sliding Window Pattern](images/sliding_window_mask_128x128.png)

### 5. Dilated Sliding Window Pattern

Each token attends to a fixed number of positions spaced by a dilation factor:
- **Diagonal self-attention (value 1)**
- **Dilated sliding window attention (value 2)**

This pattern achieves approximately **96.8% sparsity** with a window size of 32 and dilation of 2.

![Dilated Sliding Window Pattern](images/dilated_sliding_window_mask_128x128.png)

## Implementation Details

This project is implemented as a single Python file that provides:

- Mask generation functions (step-by-step implementation)
- Mask visualization tools
- Sparsity calculation and statistics
- Side-by-side comparison of different attention patterns

## Usage

### Generating Attention Masks

```python
from sparse_transformer_mask import (
    create_normal_mask_step_by_step,
    create_strided_mask_step_by_step,
    create_fixed_mask_step_by_step,
    create_sliding_window_mask_step_by_step,
    create_dilated_sliding_window_mask_step_by_step
)

# Generate normal (full) attention mask
normal_mask = create_normal_mask_step_by_step(size=1024)

# Generate strided mask
strided_mask = create_strided_mask_step_by_step(size=1024, window_size=32, stride=32)

# Generate fixed mask
fixed_mask = create_fixed_mask_step_by_step(size=1024, window_size=32)

# Generate sliding window mask
sliding_window_mask = create_sliding_window_mask_step_by_step(size=1024, window_size=32)

# Generate dilated sliding window mask
dilated_sliding_window_mask = create_dilated_sliding_window_mask_step_by_step(size=1024, window_size=32, dilation=2)
```

#### Sparsity Information in case of 1024 tokens
```
Sparsity of each mask:
Normal Mask Sparsity: 49.9512%
Strided Mask Sparsity: 95.4086%
Fixed Mask Sparsity: 96.8750%
Sliding Window Mask Sparsity: 96.8033%
Dilated Sliding Window Mask Sparsity: 96.8292%
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

### Comparing Multiple Attention Patterns

```python
from sparse_transformer_mask import visualize_mask_comparison

# Compare all five attention patterns side by side (showing 128x128 samples)
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
```

![Comparison of All Patterns](images/mask_comparison_128x128.png)

### Running the Script Directly

```bash
python mask_resource_calculator.py
```
or follow this options

```bash
% python mask_resource_calculator.py --help                           
usage: mask_resource_calculator.py [-h] [--mask_size MASK_SIZE] [--num_multiplications NUM_MULTIPLICATIONS]
                                   [--window_size WINDOW_SIZE] [--stride STRIDE] [--file] [--read_limit READ_LIMIT]

Calculate mask resources for sparse matrix multiplications.

options:
  -h, --help            show this help message and exit
  --mask_size MASK_SIZE
                        Size of the square mask matrix (default: 1024)
  --num_multiplications NUM_MULTIPLICATIONS
                        Total number of simultaneous multiplications (default: 64)
  --window_size WINDOW_SIZE
                        Size of the local attention window for strided/fixed masks (default: 32)
  --stride STRIDE       Stride for strided attention (default: 32)
  --file                Enable file output for results
  --read_limit READ_LIMIT
                        Limit the number of lines read from the file (default: 1000)
```

### File Output

When running `mask_resource_calculator.py` with the `--file` option, the script generates two types of files in the `generated/` directory:

1. Calculation Points File (`*_mask_{NUM_MULTIPLICATIONS}_read_limit_{READ_LIMIT}.txt`):
   - Contains the calculation points for each time step
   - Includes parameters used for mask generation
   - Format: `generated/{mask_type}_mask_{NUM_MULTIPLICATIONS}_read_limit_{READ_LIMIT}.txt`

2. Analysis File (`*_mask_{NUM_MULTIPLICATIONS}_read_limit_{READ_LIMIT}_analysis.txt`):
   - Contains detailed analysis of the calculation points
   - Includes maximum case information and change analysis
   - Format: `generated/{mask_type}_mask_{NUM_MULTIPLICATIONS}_read_limit_{READ_LIMIT}_analysis.txt`

Example:
```bash
# Generate and visualize attention patterns
python sparse_transformer_mask.py --size 1024 --window_size 32

# Analyze resource requirements
python mask_resource_calculator.py --file --num_multiplications 64 --read_limit 1000
```

### Mask Resource Calculator

The `mask_resource_calculator.py` script is designed to analyze the row of Q and columns of transpose(K) requirements for sparse matrix multiplications based on different attention masks. It calculates the number of unique rows and columns needed for a given number of simultaneous multiplications, helping to understand the "worst-case" spread of indices.

There are some assumptions:
- MAC array's y dimension is same as dot product counts, i.e., dimenstion or dimensions / # of head
- Adder is possible to y dimentsion of th array
- Calculate in row first manner
- Try to utilize the 100% of MAC array

To run the mask resource calculator:

```bash
python mask_resource_calculator.py --mask_size <size> --num_multiplications <num> --window_size <window> --stride <stride> [--file] [--read_limit <limit>]
```

**Arguments:**
- `--mask_size`: Size of the square mask matrix (default: 1024)
- `--num_multiplications`: Total number of simultaneous multiplications (default: 64)
- `--window_size`: Size of the local attention window for strided/fixed masks (default: 32)
- `--stride`: Stride for strided attention (default: 32)
- `--file`: Enable file output for results (optional)
- `--read_limit`: Limit the sum of unique rows and columns for point calculation (default: 1000)

**Example Usage:**

```bash
python mask_resource_calculator.py --mask_size 1024 --num_multiplications 64 --file
```
This command will calculate and print the Temp Buffer's resource requirements for a normal mask and save the detailed points to a file named `generated/normal_mask_64_read_limit_1024.txt`.

## Color Mapping

The visualization uses a custom colormap to distinguish different attention types:
- **Gray (value 0)**: No attention (masked out)
- **Dark blue (value 1)**: Diagonal/self-attention
- **Royal blue (value 2)**: Local/sliding/dilated attention
- **Sky blue (value 3)**: Strided/fixed column attention

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
