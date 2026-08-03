✅ RMSNorm Numerical Precision Verified!

--- RMSNorm Benchmark Results (Lower is Better) ---
1. PyTorch Unfused Eager : 2.4722 ms
2. PyTorch Native C++    : 0.9976 ms
3. TorchCompile (Triton): 0.7103 ms
4. Your Fused CUDA Kernel: 0.8055 ms

Speedup over Native PyTorch Eager: 1.24x
Speedup over TorchCompile:        0.88x
