import torch, my_cuda_engine_cpp
B, L, H, KV_H, D = 1, 1024, 8, 2, 128
q = torch.randn(B,L,H,D, device='cuda', dtype=torch.float16)
k = torch.randn(B,L,KV_H,D, device='cuda', dtype=torch.float16)
v = torch.randn(B,L,KV_H,D, device='cuda', dtype=torch.float16)

# 1 warmup outside profiling
my_cuda_engine_cpp.prefill_attention(q,k,v, 1.0/D**0.5)
torch.cuda.synchronize()

# ncu will start here
torch.cuda.cudart().cudaProfilerStart()
my_cuda_engine_cpp.prefill_attention(q,k,v, 1.0/D**0.5)
torch.cuda.cudart().cudaProfilerStop()
