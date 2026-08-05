import math
import time
import my_cuda_engine_cpp
import torch

from engine.cache import KVCacheManager
from engine.models.llama import Llama3BConfig, Llama3BEngineModel
from engine.runner import CUDAGraphRunner


def test_cuda_graph_execution():
    torch.manual_seed(42)
    device = "cuda"

    config_dict = {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 4,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
    }
    config = Llama3BConfig(config_dict)

    # Standard deviation scaling to prevent FP16 overflow/underflow
    std = 1.0 / math.sqrt(config.hidden_size)

    weights = {
        "model.embed_tokens.weight": torch.randn(
            (config.vocab_size, config.hidden_size), dtype=torch.float16, device=device
        ) * std,
        "model.norm.weight": torch.ones(
            (config.hidden_size,), dtype=torch.float16, device=device
        ),
        "lm_head.weight": torch.randn(
            (config.vocab_size, config.hidden_size), dtype=torch.float16, device=device
        ) * std,
    }

    for l in range(config.num_hidden_layers):
        p = f"model.layers.{l}"
        weights[f"{p}.input_layernorm.weight"] = torch.ones(
            (config.hidden_size,), dtype=torch.float16, device=device
        )
        weights[f"{p}.post_attention_layernorm.weight"] = torch.ones(
            (config.hidden_size,), dtype=torch.float16, device=device
        )
        weights[f"{p}.self_attn.q_proj.weight"] = torch.randn(
            (config.num_attention_heads * config.head_dim, config.hidden_size),
            dtype=torch.float16,
            device=device,
        ) * std
        weights[f"{p}.self_attn.k_proj.weight"] = torch.randn(
            (config.num_key_value_heads * config.head_dim, config.hidden_size),
            dtype=torch.float16,
            device=device,
        ) * std
        weights[f"{p}.self_attn.v_proj.weight"] = torch.randn(
            (config.num_key_value_heads * config.head_dim, config.hidden_size),
            dtype=torch.float16,
            device=device,
        ) * std
        weights[f"{p}.self_attn.o_proj.weight"] = torch.randn(
            (config.hidden_size, config.num_attention_heads * config.head_dim),
            dtype=torch.float16,
            device=device,
        ) * std
        weights[f"{p}.mlp.gate_proj.weight"] = torch.randn(
            (config.intermediate_size, config.hidden_size),
            dtype=torch.float16,
            device=device,
        ) * std
        weights[f"{p}.mlp.up_proj.weight"] = torch.randn(
            (config.intermediate_size, config.hidden_size),
            dtype=torch.float16,
            device=device,
        ) * std
        weights[f"{p}.mlp.down_proj.weight"] = torch.randn(
            (config.hidden_size, config.intermediate_size),
            dtype=torch.float16,
            device=device,
        ) * std

    cache_manager = KVCacheManager(
        num_blocks=100,
        block_size=16,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float16,
        device=device,
    )

    # Initialize KV cache memory with valid random data
    cache_manager.kv_cache.normal_(mean=0.0, std=std)

    model = Llama3BEngineModel(config, weights, cache_manager)
    runner = CUDAGraphRunner(model, max_batch_size=1)

    cache_manager.allocate_sequence(0, 100)

    input_ids = torch.tensor([[500]], dtype=torch.int64, device=device)
    positions = torch.tensor([[99]], dtype=torch.int32, device=device)
    context_lens = torch.tensor([100], dtype=torch.int32, device=device)
    seq_ids = [0]

    # Save clean KV snapshot
    kv_snapshot = cache_manager.kv_cache.clone()

    # 1. Capture CUDA Graph
    runner.capture_graph(input_ids, positions, seq_ids, context_lens)

    # 2. Run Graph Replay FIRST on clean cache
    cache_manager.kv_cache.copy_(kv_snapshot)
    torch.cuda.synchronize()
    logits_graph = runner.execute(input_ids, positions, seq_ids, context_lens).clone()

    # 3. Run Eager Execution SECOND on clean cache
    cache_manager.kv_cache.copy_(kv_snapshot)
    torch.cuda.synchronize()
    slot_mapping, block_tables, max_blocks = runner._prepare_inputs(
        seq_ids, context_lens
    )
    logits_eager = model.decode_step(
        input_ids, positions, context_lens, slot_mapping, block_tables, max_blocks
    ).clone()

    diff = torch.abs(logits_eager - logits_graph)
    print(f"Max Diff: {diff.max().item():.6f}, Mean Diff: {diff.mean().item():.6f}")
    print("Eager sample:", logits_eager[0, :5])
    print("Graph sample:", logits_graph[0, :5])
    print(torch.argmax(diff), diff.max())
    print(torch.nonzero(diff > 1e-2).shape)

    # Precision Check
    assert torch.allclose(logits_eager, logits_graph, atol=1e-2, rtol=1e-2), (
        "Graph replay output mismatch!"
    )
    print("✅ CUDA Graph Output Numerical Precision Verified!")

    # 4. Latency Benchmark
    iterations = 500

    for _ in range(20):
        _ = model.decode_step(
            input_ids, positions, context_lens, slot_mapping, block_tables, max_blocks
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = model.decode_step(
            input_ids, positions, context_lens, slot_mapping, block_tables, max_blocks
        )
    torch.cuda.synchronize()
    t_eager = (time.perf_counter() - start) / iterations * 1000

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = runner.execute(input_ids, positions, seq_ids, context_lens)
    torch.cuda.synchronize()
    t_graph = (time.perf_counter() - start) / iterations * 1000

    print(f"\n--- Decode Step Latency Benchmark (4-layer 3B) ---")
    print(f"Eager C++/CUDA Execution : {t_eager:.4f} ms")
    print(f"CUDA Graph Replay        : {t_graph:.4f} ms")
    print(f"Graph Speedup Ratio      : {t_eager / t_graph:.2f}x")


if __name__ == "__main__":
    test_cuda_graph_execution()
