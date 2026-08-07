import time
import math
import torch
import my_cuda_engine_cpp

def benchmark_prefill_kernel_vs_pytorch():
    device = "cuda"
    dtype = torch.float16
    
    batch_size = 1
    num_heads = 24
    num_kv_heads = 8
    head_dim = 128
    hidden_size = num_heads * head_dim
    
    seq_lengths = [128, 256, 512, 1024, 2048]
    scale = 1.0 / math.sqrt(head_dim)

    print("==================================================================================")
    print(" 🚀 PREFILL ATTENTION KERNEL BENCHMARK: Custom CUDA vs PyTorch FlashAttention")
    print("==================================================================================")
    print(f"{'Seq Len':<10} | {'Custom CUDA (ms)':<18} | {'PyTorch SDPA (ms)':<18} | {'Speedup':<10}")
    print("----------------------------------------------------------------------------------")

    for seq_len in seq_lengths:
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # PyTorch SDPA format: [batch, heads, seq, dim]
        q_pt = q.transpose(1, 2)
        k_pt = k.transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_pt = v.transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)

        # Warmup
        for _ in range(10):
            _ = my_cuda_engine_cpp.prefill_attention(q, k, v, scale)
            _ = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True, scale=scale)
        torch.cuda.synchronize()

        # Benchmark Custom CUDA Kernel
        iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = my_cuda_engine_cpp.prefill_attention(q, k, v, scale)
        torch.cuda.synchronize()
        t_custom = (time.perf_counter() - t0) / iters * 1000.0

        # Benchmark PyTorch SDPA
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True, scale=scale)
        torch.cuda.synchronize()
        t_pt = (time.perf_counter() - t0) / iters * 1000.0

        speedup = t_pt / t_custom
        print(f"{seq_len:<10} | {t_custom:<18.4f} | {t_pt:<18.4f} | {speedup:.2f}x")

    print("==================================================================================")

if __name__ == "__main__":
    benchmark_prefill_kernel_vs_pytorch()
