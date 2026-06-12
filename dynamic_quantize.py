import torch
import triton
import triton.language as tl

# Dynamic Token-Wise Quantization for LLM Activations
@triton.jit
def _token_wise_quantize_kernel(x_ptr, y_ptr, scales_ptr, stride_x_token, stride_x_dim, stride_y_token, stride_y_dim, hidden_dim, BLOCK_SIZE: tl.constexpr):

    token_idx = tl.program_id(0)
    
    # Compute the memory pointers for this specific token's row
    row_start_x = x_ptr + token_idx * stride_x_token
    row_start_y = y_ptr + token_idx * stride_y_token
    
    # Create offsets for the hidden dimension
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    # Load the token's activation row from global memory
    x = tl.load(row_start_x + offsets * stride_x_dim, mask=mask, other=0.0)
    
    # Compute token-wise absolute maximum
    max_val = tl.max(tl.abs(x), axis=0)
    
    # Similarly calculate the scale and clamp it to prevent division by zero
    scale = max_val / 127.0
    scale = tl.where(scale < 1e-9, 1e-9, scale)
    
    # Similarly quantize using hardware-aligned banker's rounding
    y_fp = x / scale
    y_int8 = tl.extra.cuda.libdevice.rint(y_fp).to(tl.int8)
    
    # Similarly, store the quantized values and scales back to global memory
    tl.store(row_start_y + offsets * stride_y_dim, y_int8, mask=mask)
    tl.store(scales_ptr + token_idx, scale)


# wrapper for dynamic quantization of activations
def dynamic_quantize_activations(x: torch.Tensor):

    # Expecting x with shape of (num_tokens, hidden_dim)
    assert x.dim() == 2, "Input must be 2D (num_tokens, hidden_dim)"
    num_tokens, hidden_dim = x.shape
    
    y = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)
    
    # (Triton requirment): Find the next power of 2 for the block size 
    block_size = triton.next_power_of_2(hidden_dim)
    
    grid = lambda meta: (num_tokens,)
    
    _token_wise_quantize_kernel[grid](
        x, y, scales,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        hidden_dim,
        BLOCK_SIZE=block_size
    )
    return y, scales

def test_dynamic_quantization():

    print("Testing Dynamic Token-Wise Quantization...")
    torch.manual_seed(42)
    
    # similar simulation as in block_quantize.py
    num_tokens, hidden_dim = 128, 4096 
    
    x = torch.randn((num_tokens, hidden_dim), device='cuda', dtype=torch.float32)
    
    print(f"Activation shape: {num_tokens} tokens, {hidden_dim} dimensions.")
    triton_y, triton_scales = dynamic_quantize_activations(x)
    
    # PyTorch Baseline (Row-wise)
    pt_max = torch.max(torch.abs(x), dim=1)[0]
    pt_scales = torch.clamp(pt_max / 127.0, min=1e-9)
    pt_y = torch.round(x / pt_scales.unsqueeze(1)).to(torch.int8)
    
    # Validation
    diff = torch.abs(triton_y.to(torch.int32) - pt_y.to(torch.int32))
    max_diff = diff.max().item()
    
    if max_diff <= 1:
        print("Dynamic Activation Quantization matches PyTorch (max diff <= 1)!")
    else:
        print(f"Max diff is {max_diff}")

if __name__ == "__main__":
    test_dynamic_quantization()