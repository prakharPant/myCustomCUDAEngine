import math
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import my_cuda_engine_cpp

def measure_gpu_time_ms(func, iters=100):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        func()
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iters):
        func()
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / iters

def check_correctness(out_custom, out_pt, atol=1e-2, rtol=1e-2):
    max_diff = (out_custom - out_pt).abs().max().item()
    mean_diff = (out_custom - out_pt).abs().mean().item()
    is_correct = torch.allclose(out_custom, out_pt, atol=atol, rtol=rtol)
    return is_correct, max_diff, mean_diff

def benchmark_prefill_kernel_vs_pytorch():
    device = "cuda"
    dtype = torch.float16
    
    batch_size = 1
    num_heads = 24
    num_kv_heads = 8
    head_dim = 128
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    scale = 1.0 / math.sqrt(head_dim)

    # Detect SDPA Native GQA support
    supports_native_gqa = False
    try:
        dummy_q = torch.randn(1, 24, 64, 128, dtype=dtype, device=device)
        dummy_k = torch.randn(1, 8, 64, 128, dtype=dtype, device=device)
        _ = torch.nn.functional.scaled_dot_product_attention(dummy_q, dummy_k, dummy_k, is_causal=True, enable_gqa=True)
        supports_native_gqa = True
    except Exception:
        supports_native_gqa = False

    print("====================================================================================================")
    print(f" 🚀 PREFILL ATTENTION BENCHMARK (PyTorch Native GQA Support: {supports_native_gqa})")
    print("====================================================================================================")
    print(f"{'Seq Len':<8} | {'Status':<6} | {'Max Diff':<10} | {'Custom CUDA (ms)':<16} | {'PyTorch SDPA (ms)':<18} | {'Speedup':<8}")
    print("----------------------------------------------------------------------------------------------------")

    for seq_len in seq_lengths:
        # Custom format: [batch, seq, num_heads, head_dim]
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

        # Transpose once outside timing loops
        q_pt = q.transpose(1, 2)
        
        if supports_native_gqa:
            k_pt = k.transpose(1, 2)
            v_pt = v.transpose(1, 2)
            sdpa_fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                q_pt, k_pt, v_pt, is_causal=True, scale=scale, enable_gqa=True
            )
        else:
            # Pre-expand K/V OUTSIDE timing loop to isolate pure kernel execution time
            k_pt = k.transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
            v_pt = v.transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
            sdpa_fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                q_pt, k_pt, v_pt, is_causal=True, scale=scale
            )

        custom_fn = lambda: my_cuda_engine_cpp.prefill_attention(q, k, v, scale)

        # 1. Numerical Correctness Check
        out_custom = custom_fn()
        out_pt = sdpa_fn().transpose(1, 2) # Align layout for comparison
        is_correct, max_diff, mean_diff = check_correctness(out_custom, out_pt)
        status = "PASSED" if is_correct else "FAILED"

        # 2. CUDA Event Benchmarking (Pure GPU execution time)
        t_custom = measure_gpu_time_ms(custom_fn, iters=100)
        t_pt = measure_gpu_time_ms(sdpa_fn, iters=100)

        speedup = t_pt / t_custom
        print(f"{seq_len:<8} | {status:<6} | {max_diff:<10.4e} | {t_custom:<16.4f} | {t_pt:<18.4f} | {speedup:.2f}x")

    print("====================================================================================================")

if __name__ == "__main__":
    benchmark_prefill_kernel_vs_pytorch()
