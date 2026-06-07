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


## References

[1] Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. Advances in Neural Information Processing Systems.

[2] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023).OPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. International Conference on Learning Representations.

[3] [OpenAI Triton Documentation](https://triton-lang.org/main/index.html)

