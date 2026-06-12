# Block-Wise-Absmax-Quantization-Kernel

A custom OpenAI Triton kernel implementing block-wise absolute maximum quantization (FP32 -> INT8), verified against a native PyTorch implementation. 

## Context & Motivation
This repository was built over a single weekend as a focused translation of my existing GPU-accelerated computing experience into the Triton ecosystem. 

My previous research has heavily utilized CuPy and custom CUDA matrix vectorizations to eliminate CPU bottlenecks in highly parallelized evolutionary algorithms (e.g., accelerating causal discovery mechanisms). As I pivot my focus toward Efficient AI and model compression, I built this kernel to familiarize myself with Triton's hardware-level abstractions (SRAM management, pointer arithmetic, and block-level reductions) for quantization workflows.

## The Kernel
The `_block_absmax_quantize_kernel` performs a foundational operation common in LLM parameter quantization (similar to the mechanics behind `LLM.int8()`):
1. **Memory Coalescing**: Divides a flattened 1D tensor into contiguous chunks of `BLOCK_SIZE` to maximize global memory bandwidth.
2. **On-Chip Reduction**: Loads blocks into SRAM to calculate the absolute maximum value locally.
3. **Quantization**: Applies, rounds and casts a scale factor mapping to the derived [-127, 127]` INT8 range.
4. **Write-back**: Stores the compressed INT8 tensor and the individual FP32 scale factors back to High Bandwidth Memory (HBM).


## Dynamic Token-Wise Activation Quantization (Updated on 12 June 2026)
  `dynamic_quantize.py` treats data as a 2D matrix shape of (num_tokens, hidden_dim) where each GPU thread block processes exactly one token instead of a flat, static array as in `block_quantize.py`. The main motivation is because the hidden states and user inputs will change in every forward pass, such that a token-wise quantization in the GPU allows 2D strided memory access in Triton before they can be multiplied with the weights.

## Results

| Size | Triton Kernel (GB/s) | PyTorch Native (GB/s) |
|------:|---------------------:|----------------------:|
| 65,536 | 30,117.65 | 6,664.50 |
| 131,072 | 45,816.55 | 21,355.58 |
| 262,144 | 108,359.79 | 33,684.21 |
| 524,288 | 147,870.03 | 40,524.36 |
| 1,048,576 | 166,335.02 | 32,003.13 |
| 2,097,152 | 182,449.88 | 32,665.11 |
| 4,194,304 | 192,244.05 | 33,738.84 |
| 8,388,608 | 191,878.20 | 34,294.54 |
| 16,777,216 | 181,201.35 | 34,352.06 |
| 33,554,432 | 170,705.56 | 34,246.37 |


<img width="605" height="433" alt="image" src="https://github.com/user-attachments/assets/db838916-a3a6-41b3-998e-870c366dcd97" />

At 4,194,304 elements, the speed up using Triton kernel is around ~5.7x. Interstingly, based on the plot the Triton kernel climbs rapidly and peaks around 192,244 (at size ~4M) then slightly plateaus at ~170,705 for massive 33M size. This indicates that the GPU's compute cores are no longer the botteneck for extremely large matrix.

## References

[1] Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. Advances in Neural Information Processing Systems.

[2] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023).OPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. International Conference on Learning Representations.

[3] [OpenAI Triton Documentation](https://triton-lang.org/main/index.html)

