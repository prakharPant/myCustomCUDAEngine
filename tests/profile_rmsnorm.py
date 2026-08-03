import torch
import my_cuda_engine_cpp

hidden_size = 4096
x = torch.randn(8, 2048, hidden_size, dtype=torch.float16, device="cuda")
w = torch.ones(hidden_size, dtype=torch.float16, device="cuda")

# 10 warmup - these will be skipped
for _ in range(10):
    my_cuda_engine_cpp.rmsnorm(x, w, 1e-5)
torch.cuda.synchronize()

# --- this is the ONE we will profile ---
my_cuda_engine_cpp.rmsnorm(x, w, 1e-5)
torch.cuda.synchronize()
