import torch
import torch.nn as nn
import time
import my_cuda_engine_cpp

class PythonUnfusedRMSNorm(nn.Module):
    """Unfused math ops (What you previously benchmarked)"""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16, device="cuda"))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def benchmark_rmsnorm():
    batch_size, seq_len, hidden_size = 8, 2048, 4096
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device="cuda")
    
    # Baseline 1: Unfused Python Eager
    unfused_norm = PythonUnfusedRMSNorm(hidden_size)
    
    # Baseline 2: PyTorch Native C++ Eager (Fused internally in PyTorch 2.x+)
    native_norm = nn.RMSNorm(hidden_size, eps=1e-5, device="cuda", dtype=torch.float16)

    # Baseline 3: TorchCompile (Generates a fused Triton kernel automatically)
    compiled_norm = torch.compile(unfused_norm)
    # Warmup torch.compile execution graph
    _ = compiled_norm(x)

    # 1. Verification
    out_native = native_norm(x)
    out_custom = my_cuda_engine_cpp.rmsnorm(x, native_norm.weight, 1e-5)
    assert torch.allclose(out_native, out_custom, atol=1e-2, rtol=1e-2)
    print("✅ RMSNorm Numerical Precision Verified!")

    # Benchmark function
    def measure_time(fn, inputs, iterations=1000):
        # Warmup GPU
        for _ in range(50): fn(*inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn(*inputs)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iterations * 1000 # returns ms

    t_unfused = measure_time(unfused_norm, (x,))
    t_native = measure_time(native_norm, (x,))
    t_compiled = measure_time(compiled_norm, (x,))
    t_custom = measure_time(my_cuda_engine_cpp.rmsnorm, (x, native_norm.weight, 1e-5))

    print(f"\n--- RMSNorm Benchmark Results (Lower is Better) ---")
    print(f"1. PyTorch Unfused Eager : {t_unfused:.4f} ms")
    print(f"2. PyTorch Native C++    : {t_native:.4f} ms")
    print(f"3. TorchCompile (Triton): {t_compiled:.4f} ms")
    print(f"4. Your Fused CUDA Kernel: {t_custom:.4f} ms")
    
    print(f"\nSpeedup over Native PyTorch Eager: {t_native / t_custom:.2f}x")
    print(f"Speedup over TorchCompile:        {t_compiled / t_custom:.2f}x")

if __name__ == "__main__":
    benchmark_rmsnorm()
