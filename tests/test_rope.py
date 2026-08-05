import torch
import time
import my_cuda_engine_cpp

def precompute_cos_sin_tables(max_seq_len: int, head_dim: int, base: float = 10000.0, device: str = "cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def pytorch_apply_rope(q, k, cos, sin, positions):
    # Standard Llama half-half rotation
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Gather cos/sin for current positions
    cos_pos = cos[positions].unsqueeze(2).to(q.dtype) # [batch, seq, 1, head_dim/2]
    sin_pos = sin[positions].unsqueeze(2).to(q.dtype)
    
    cos_pos = torch.cat([cos_pos, cos_pos], dim=-1)
    sin_pos = torch.cat([sin_pos, sin_pos], dim=-1)

    q_embed = (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_embed = (k * cos_pos) + (rotate_half(k) * sin_pos)
    return q_embed, k_embed

def test_rope():
    batch_size, seq_len, num_heads, num_kv_heads, head_dim = 16, 1, 32, 8, 128
    max_seq_len = 4096

    cos_table, sin_table = precompute_cos_sin_tables(max_seq_len, head_dim)
    positions = torch.randint(0, 2048, (batch_size, seq_len), dtype=torch.int32, device="cuda")

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

    # PyTorch Reference
    q_ref, k_ref = pytorch_apply_rope(q.clone(), k.clone(), cos_table, sin_table, positions)

    q_ref = q_ref.half()
    k_ref = k_ref.half()

    # Custom In-Place Kernel Execution
    q_custom = q.clone()
    k_custom = k.clone()
    my_cuda_engine_cpp.rope_inplace(q_custom, k_custom, cos_table, sin_table, positions)

    # Precision check
    assert torch.allclose(q_ref, q_custom, atol=1e-2, rtol=1e-2), "Query RoPE precision mismatch!"
    assert torch.allclose(k_ref, k_custom, atol=1e-2, rtol=1e-2), "Key RoPE precision mismatch!"
    print("✅ Fused In-Place RoPE Precision Verified!")

    # Benchmark
    def measure(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(1000): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start)

    t_ref = measure(lambda: pytorch_apply_rope(q, k, cos_table, sin_table, positions))
    t_custom = measure(lambda: my_cuda_engine_cpp.rope_inplace(q_custom, k_custom, cos_table, sin_table, positions))

    print(f"\nPyTorch RoPE Latency: {t_ref * 1000:.4f} ms")
    print(f"Fused CUDA RoPE Latency: {t_custom * 1000:.4f} ms")
    print(f"Speedup: {t_ref / t_custom:.2f}x")

if __name__ == "__main__":
    test_rope()
