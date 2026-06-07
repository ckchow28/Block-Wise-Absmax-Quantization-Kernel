import torch
import triton
import triton.language as tl


# TRITON KERNEL: Block-Wise Absmax Quantization
@triton.jit
def _block_absmax_quantize_kernel(x_ptr, y_ptr, scales_ptr,n_elements,BLOCK_SIZE: tl.constexpr):
  
    # Identify which block this program instance is responsible for
    pid = tl.program_id(axis=0)
    
    # Calculate the starting memory 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to prevent out-of-bounds memory access on the final block
    mask = offsets < n_elements
    
    # load data frm global memory
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0) # Note: padding with 0.0 for unmapped elements
    
    # Compute the absolute maximum value in the block
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    
    # Calculate the quantization scale
    scale = max_val / 127.0  # 2^7 -1 = 127
    # scale = tl.where(scale == 0.0, 1e-9, scale)
    scale = tl.where(scale < 1e-9, 1e-9, scale) # prevent division by zero
    
    # Quantize values
    y_fp = x / scale # divide by scale
    # y_rounded = tl.extra.cuda.libdevice.round(y_fp) 
    y_rounded = tl.extra.cuda.libdevice.rint(y_fp) # IMPORTANT: Use rint for rounding to nearest integer (ties to even) to match PyTorch's behavior
    y_int8 = y_rounded.to(tl.int8) # cast to int8
    
    # Store quantized values and the scale factor back to global memory
    tl.store(y_ptr + offsets, y_int8, mask=mask)
    tl.store(scales_ptr + pid, scale)


# wrapper
def block_quantize(x: torch.Tensor, block_size: int = 128):

    assert x.is_contiguous(), "Input tensor must be contiguous in memory."
    n_elements = x.numel()
    
    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.int8)
    
    # Grid calculation: how many blocks do we need? 
    n_blocks = triton.cdiv(n_elements, block_size)  # Ceiling division
    scales = torch.empty((n_blocks,), device=x.device, dtype=torch.float32)
    
    # Launch the kernel
    grid = lambda meta: (n_blocks,)
    
    _block_absmax_quantize_kernel[grid](
        x, y, scales, n_elements,
        BLOCK_SIZE=block_size
    )
    
    return y, scales


# PyTorch comparison
def test_quantization():
    torch.manual_seed(42)
    n_elements = 4096 * 4096  # its just a typical weight matrix simulation
    block_size = 128
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    x = torch.randn(n_elements, device=device, dtype=torch.float32)
    
    # Run custom Triton Kernel
    print(f"Running Triton block quantization (Block Size: {block_size}):")
    triton_y, triton_scales = block_quantize(x, block_size=block_size)
    
    # Run Pure PyTorch 
    print("Running PyTorch baseline:")
    x_blocks = x.view(-1, block_size)

    # Find max absolute value per block
    pt_max_vals = torch.max(torch.abs(x_blocks), dim=1)[0]
    pt_scales = pt_max_vals / 127.0
    pt_scales = torch.clamp(pt_scales, min=1e-9)
    
    # Quantize
    pt_y = torch.round(x_blocks / pt_scales.unsqueeze(1)).to(torch.int8).view(-1)
    
    print("Outputs: Triton vs PyTorch")
    scales_match = torch.allclose(triton_scales, pt_scales, rtol=1e-4, atol=1e-4)
    
    # Calculate exact INT8 differences 
    diff = torch.abs(triton_y.to(torch.int32) - pt_y.to(torch.int32))
    max_diff = diff.max().item()
    mismatches = (diff > 0).sum().item()
    
    print(f"Max INT8 difference: {max_diff}")
    print(f"Total mismatched elements: {mismatches} out of {n_elements} ({(mismatches/n_elements)*100:.4f}%)")
    
    # IMPORTANT: Accept tolerance of 1 for hardware FP32 division discrepancies
    y_match = max_diff <= 1
    
    if scales_match and y_match:
        print("Triton kernel matches PyTorch baseline (within expected FP32 hardware tolerance)!")
    else:
        print("Kernel output diverges from PyTorch beyond tolerance.")
        print(f"Scales match: {scales_match}")

if __name__ == "__main__":
    test_quantization()