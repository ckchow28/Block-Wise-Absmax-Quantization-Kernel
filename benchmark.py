import torch
import triton
from block_quantize import block_quantize

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  
        x_vals=[2**i for i in range(16, 26)],  # Tensor sizes from ~65K to ~33M elements
        line_arg='provider',  
        line_vals=['triton', 'pytorch'],  
        line_names=['Triton Kernel', 'PyTorch Native'],  
        styles=[('blue', '-'), ('green', '--')],  
        ylabel='GB/s',  
        plot_name='quantization-performance',  
        args={'block_size': 128},  
    )
)
def benchmark(size, provider, block_size):
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: block_quantize(x, block_size), quantiles=quantiles)
    
    if provider == 'pytorch':
        def pt_baseline(t, b_size):
            t_blocks = t.view(-1, b_size)
            scales = torch.max(torch.abs(t_blocks), dim=1)[0] / 127.0
            scales = torch.clamp(scales, min=1e-9)
            return torch.round(t_blocks / scales.unsqueeze(1)).to(torch.int8)
            
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: pt_baseline(x, block_size), quantiles=quantiles)
    
    # Calculate the memory throughput 
    # just read 1 FP32 tensor (4 bytes) and write 1 INT8 tensor (1 byte) + 1 FP32 scale per block
    gbps = lambda ms: (5 * x.numel()) / (ms * 1e-6) / 1e9
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    benchmark.run(print_data=True, save_path='.')