import torch
import math
import random
import time
import my_cuda_engine_cpp
from engine.cache import KVCacheManager

def benchmark_and_verify_paged_attention():
    torch.manual_seed(42)
    random.seed(42)

    # Llama 3 8B-like decode setup
    batch_size = 16
    num_heads = 32
    num_kv_heads = 32
    head_dim = 128
    block_size = 16
    context_len = 2048  # Realistic context length during decode phase
    scale = 1.0 / math.sqrt(head_dim)

    context_lens = [context_len] * batch_size
    context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, device="cuda")

    # Memory pool setup
    total_blocks = (context_len // block_size) * batch_size + 100
    cache_manager = KVCacheManager(
        num_blocks=total_blocks,
        block_size=block_size,
        num_layers=1,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda"
    )

    key_cache = torch.zeros((total_blocks, block_size, num_kv_heads, head_dim), dtype=torch.float16, device="cuda")
    value_cache = torch.zeros((total_blocks, block_size, num_kv_heads, head_dim), dtype=torch.float16, device="cuda")

    query = torch.randn((batch_size, num_heads, head_dim), dtype=torch.float16, device="cuda")

    # Contiguous KV tensors for PyTorch SDPA comparison
    k_contiguous = torch.randn((batch_size, context_len, num_kv_heads, head_dim), dtype=torch.float16, device="cuda")
    v_contiguous = torch.randn((batch_size, context_len, num_kv_heads, head_dim), dtype=torch.float16, device="cuda")

    # Populate Paged Cache from contiguous data
    for i in range(batch_size):
        cache_manager.allocate_sequence(seq_id=i, num_tokens=context_len)
        block_table = cache_manager.block_tables[i]
        
        for token_idx in range(context_len):
            b_idx = block_table[token_idx // block_size]
            b_offset = token_idx % block_size
            key_cache[b_idx, b_offset] = k_contiguous[i, token_idx]
            value_cache[b_idx, b_offset] = v_contiguous[i, token_idx]

    max_blocks_per_seq = context_len // block_size
    block_tables_tensor = cache_manager.get_block_table_tensor(
        seq_ids=list(range(batch_size)), 
        max_blocks_per_seq=max_blocks_per_seq
    )

    # -------------------------------------------------------------
    # 1. PRECISION CHECK
    # -------------------------------------------------------------
    # Shape targets: [batch_size, num_heads, seq_len, head_dim]
    q_sdpa = query.unsqueeze(2)                 # [16, 32, 1, 128]
    k_sdpa = k_contiguous.permute(0, 2, 1, 3)   # [16, 32, 2048, 128]
    v_sdpa = v_contiguous.permute(0, 2, 1, 3)   # [16, 32, 2048, 128]

    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, scale=scale
    ).squeeze(2) # Output shape: [16, 32, 128]

    custom_out = my_cuda_engine_cpp.paged_attention_decode(
        query,
        key_cache,
        value_cache,
        block_tables_tensor,
        context_lens_tensor,
        max_blocks_per_seq,
        block_size,
        scale
    )

    assert ref_out.shape == custom_out.shape, f"Shape mismatch! Ref: {ref_out.shape}, Custom: {custom_out.shape}"
    assert torch.allclose(ref_out, custom_out, atol=1e-2, rtol=1e-2), \
        f"Precision failure! Max diff: {(ref_out - custom_out).abs().max()}"
    print("✅ Paged FlashAttention Precision Verified!")

    # -------------------------------------------------------------
    # 2. PERFORMANCE BENCHMARK
    # -------------------------------------------------------------
    iterations = 500

    # Warmup
    for _ in range(50):
        _ = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=scale)
        _ = my_cuda_engine_cpp.paged_attention_decode(
            query, key_cache, value_cache, block_tables_tensor, 
            context_lens_tensor, max_blocks_per_seq, block_size, scale
        )

    # Benchmark PyTorch SDPA (Contiguous memory)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=scale)
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / iterations * 1000

    # Benchmark Custom Paged FlashAttention (Non-contiguous memory)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = my_cuda_engine_cpp.paged_attention_decode(
            query, key_cache, value_cache, block_tables_tensor, 
            context_lens_tensor, max_blocks_per_seq, block_size, scale
        )
    torch.cuda.synchronize()
    custom_time = (time.perf_counter() - start) / iterations * 1000

    print(f"\n--- Decode Attention Benchmark (Batch={batch_size}, Context={context_len}) ---")
    print(f"PyTorch SDPA (Contiguous Memory): {sdpa_time:.4f} ms")
    print(f"Custom Paged Attention (Block Memory): {custom_time:.4f} ms")
    print(f"Performance Ratio: {sdpa_time / custom_time:.2f}x relative to PyTorch SDPA")

if __name__ == "__main__":
    benchmark_and_verify_paged_attention()
