import time
import math
import gc
import torch
import my_cuda_engine_cpp
from typing import List
from engine.cache import KVCacheManager
from engine.models.llama import Llama3BConfig, Llama3BEngineModel
from engine.runner import CUDAGraphRunner


def prefill_step(
    model: Llama3BEngineModel,
    input_ids: torch.Tensor,      # [batch_size, seq_len]
    seq_ids: List[int],
    context_lens: torch.Tensor
) -> torch.Tensor:
    """
    Executes a complete FP16 Prefill pass for Llama-3.2-3B:
    RMSNorm -> QKV Projections -> RoPE -> Paged KV Store -> Prefill Attention ->
    O Projection -> Residual -> RMSNorm -> SwiGLU MLP -> Residual -> Final Norm & LM Head.
    """
    batch_size, seq_len = input_ids.shape
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    # Initial Embedding Lookup
    x = model.weights["model.embed_tokens.weight"][input_ids]  # [batch_size, seq_len, hidden_size]

    for l in range(model.config.num_hidden_layers):
        layer_prefix = f"model.layers.{l}"
        residual = x

        # ---------------------------------------------------------------------
        # 1. Input Layernorm & QKV Projections
        # ---------------------------------------------------------------------
        norm_w1 = model.weights[f"{layer_prefix}.input_layernorm.weight"]
        x_flat = x.view(-1, hidden_size)
        x_norm1 = my_cuda_engine_cpp.rmsnorm(x_flat, norm_w1, model.config.rms_norm_eps).view(batch_size, seq_len, hidden_size)

        w_q = model.weights[f"{layer_prefix}.self_attn.q_proj.weight"]
        w_k = model.weights[f"{layer_prefix}.self_attn.k_proj.weight"]
        w_v = model.weights[f"{layer_prefix}.self_attn.v_proj.weight"]

        q = torch.matmul(x_norm1, w_q.T).view(batch_size, seq_len, num_heads, head_dim)
        k = torch.matmul(x_norm1, w_k.T).view(batch_size, seq_len, num_kv_heads, head_dim)
        v = torch.matmul(x_norm1, w_v.T).view(batch_size, seq_len, num_kv_heads, head_dim)

        # ---------------------------------------------------------------------
        # 2. Rotary Position Embeddings (RoPE) In-Place
        # ---------------------------------------------------------------------
        pos_range = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0).repeat(batch_size, 1)
        my_cuda_engine_cpp.rope_inplace(q, k, model.cos_table, model.sin_table, pos_range)

        # ---------------------------------------------------------------------
        # 3. Store K/V into Paged KV Cache Pool
        # ---------------------------------------------------------------------
        key_cache = model.cache_manager.kv_cache[l, 0]
        val_cache = model.cache_manager.kv_cache[l, 1]

        for p in range(seq_len):
            slot_mapping = torch.zeros((batch_size,), dtype=torch.int32, device="cuda")
            for i, seq_id in enumerate(seq_ids):
                b_idx = model.cache_manager.block_tables[seq_id][p // model.cache_manager.block_size]
                b_off = p % model.cache_manager.block_size
                slot_mapping[i] = b_idx * model.cache_manager.block_size + b_off

            my_cuda_engine_cpp.paged_kv_store(
                k[:, p], v[:, p], key_cache, val_cache, slot_mapping
            )

        # ---------------------------------------------------------------------
        # 4. Prefill Sequence Attention (GQA-compatible SDPA with Causal Mask)
        # ---------------------------------------------------------------------
        # Reshape to [batch_size, heads, seq_len, head_dim] for PyTorch SDPA
        q_attn = q.transpose(1, 2)
        k_attn = k.transpose(1, 2)
        v_attn = v.transpose(1, 2)

        # Expand K/V heads for Grouped Query Attention if needed
        if num_heads != num_kv_heads:
            num_queries_per_kv = num_heads // num_kv_heads
            k_attn = k_attn.repeat_interleave(num_queries_per_kv, dim=1)
            v_attn = v_attn.repeat_interleave(num_queries_per_kv, dim=1)

        attn_out = my_cuda_engine_cpp.prefill_attention(q, k, v, scale)
        attn_out = attn_out.view(batch_size, seq_len, hidden_size)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        # Output Projection & Attention Residual Add
        w_o = model.weights[f"{layer_prefix}.self_attn.o_proj.weight"]
        x = residual + torch.matmul(attn_out, w_o.T)

        # ---------------------------------------------------------------------
        # 5. Post-Attention Layernorm & SwiGLU MLP Block
        # ---------------------------------------------------------------------
        residual = x
        norm_w2 = model.weights[f"{layer_prefix}.post_attention_layernorm.weight"]
        x_norm2 = my_cuda_engine_cpp.rmsnorm(x.view(-1, hidden_size), norm_w2, model.config.rms_norm_eps)

        w_gate = model.weights[f"{layer_prefix}.mlp.gate_proj.weight"]
        w_up = model.weights[f"{layer_prefix}.mlp.up_proj.weight"]
        w_down = model.weights[f"{layer_prefix}.mlp.down_proj.weight"]

        # Compute Gate and Up projections
        gate = torch.matmul(x_norm2, w_gate.T)
        up = torch.matmul(x_norm2, w_up.T)

        # Concatenate along final dim for fused SwiGLU C++/CUDA kernel
        gate_up = torch.cat([gate, up], dim=-1)
        mlp_intermediate = my_cuda_engine_cpp.swiglu(gate_up)

        # Down projection & MLP Residual Add
        mlp_out = torch.matmul(mlp_intermediate, w_down.T).view(batch_size, seq_len, hidden_size)
        x = residual + mlp_out

    # -------------------------------------------------------------------------
    # 6. Final RMSNorm & LM Head Projection (Last Token Logits)
    # -------------------------------------------------------------------------
    final_norm_w = model.weights["model.norm.weight"]
    final_norm = my_cuda_engine_cpp.rmsnorm(x[:, -1], final_norm_w, model.config.rms_norm_eps)

    lm_head_w = model.weights.get("lm_head.weight", model.weights["model.embed_tokens.weight"])
    logits = torch.matmul(final_norm, lm_head_w.T)

    return logits


def run_benchmark():
    device = "cuda"
    print("🚀 Starting Complete End-to-End Latency Benchmark for Llama-3.2-3B Architecture...")

    config_dict = {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,  # Full 28 layers
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5
    }
    config = Llama3BConfig(config_dict)

    # Standard deviation scaling for synthetic weights to avoid overflow
    std_hidden = 1.0 / math.sqrt(config.hidden_size)
    std_inter = 1.0 / math.sqrt(config.intermediate_size)

    # Initialize Cache Manager
    block_size = 16
    cache_manager = KVCacheManager(
        num_blocks=256,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float16,
        device=device
    )

    # Synthetic Weight Population (Full Layer Set)
    print("Allocating Llama-3.2-3B FP16 Weight Tensors...")

    # 2. Embed Tokens Weight
    embed_weight = torch.randn((config.vocab_size, config.hidden_size), dtype=torch.float16, device=device) * std_hidden

    weights = {
        "model.embed_tokens.weight": embed_weight,
        "model.norm.weight": torch.ones((config.hidden_size,), dtype=torch.float16, device=device),
        "lm_head.weight": embed_weight
    }

    # 3. Layer Weight Allocation under torch.inference_mode()
    with torch.inference_mode():
      for l in range(config.num_hidden_layers):
          p = f"model.layers.{l}"
          weights[f"{p}.input_layernorm.weight"] = torch.ones((config.hidden_size,), dtype=torch.float16, device=device)
          weights[f"{p}.post_attention_layernorm.weight"] = torch.ones((config.hidden_size,), dtype=torch.float16, device=device)
          weights[f"{p}.self_attn.q_proj.weight"] = torch.randn((config.num_attention_heads * config.head_dim, config.hidden_size), dtype=torch.float16, device=device) * std_hidden
          weights[f"{p}.self_attn.k_proj.weight"] = torch.randn((config.num_key_value_heads * config.head_dim, config.hidden_size), dtype=torch.float16, device=device) * std_hidden
          weights[f"{p}.self_attn.v_proj.weight"] = torch.randn((config.num_key_value_heads * config.head_dim, config.hidden_size), dtype=torch.float16, device=device) * std_hidden
          weights[f"{p}.self_attn.o_proj.weight"] = torch.randn((config.hidden_size, config.num_attention_heads * config.head_dim), dtype=torch.float16, device=device) * std_hidden
          weights[f"{p}.mlp.gate_proj.weight"] = torch.randn((config.intermediate_size, config.hidden_size), dtype=torch.float16, device=device) * std_hidden
          weights[f"{p}.mlp.up_proj.weight"] = torch.randn((config.intermediate_size, config.hidden_size), dtype=torch.float16, device=device) * std_hidden
          weights[f"{p}.mlp.down_proj.weight"] = torch.randn((config.hidden_size, config.intermediate_size), dtype=torch.float16, device=device) * std_inter

    # Force Garbage Collection before warm-up
    gc.collect()
    torch.cuda.empty_cache()


    model = Llama3BEngineModel(config, weights, cache_manager)
    runner = CUDAGraphRunner(model, max_batch_size=1)

    # Benchmark Parameters
    prompt_len = 512
    generate_tokens = 128
    batch_size = 1
    seq_ids = [0]

    # Pre-allocate sequence in Block Manager
    cache_manager.allocate_sequence(0, prompt_len + generate_tokens)

    # Warmup prefill
    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), dtype=torch.int64, device=device)
    context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)
    _ = prefill_step(model, prompt_ids, seq_ids, context_lens)
    torch.cuda.synchronize()

    # --- 1. Measure Time to First Token (TTFT - Prefill Phase) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = prefill_step(model, prompt_ids, seq_ids, context_lens)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n--- Benchmark Results ---")
    print(f"Prompt Length          : {prompt_len} tokens")
    print(f"Time To First Token    : {ttft_ms:.2f} ms")

    # --- 2. Measure Inter-Token Latency (ITL - Autoregressive Decode Phase) ---
    decode_times = []

    # Compute static MAX blocks for CUDA Graph shape invariance
    max_seq_len = prompt_len + generate_tokens
    max_blocks = (max_seq_len + block_size - 1) // block_size

    next_input_id = torch.tensor([[101]], dtype=torch.int64, device=device)
    next_pos = torch.tensor([[prompt_len]], dtype=torch.int32, device=device)
    curr_context_len = torch.tensor([prompt_len + 1], dtype=torch.int32, device=device)

    slot_mapping, block_tables_tensor, _ = runner._prepare_inputs(
        seq_ids, curr_context_len
    )

    # Warmup graph capture pass
    runner.capture_graph(
        next_input_id, next_pos, seq_ids, curr_context_len
    )

    print(f"Generating {generate_tokens} tokens...")
    for i in range(generate_tokens):
        pos_val = prompt_len + i
        next_pos.copy_(torch.tensor([[pos_val]], dtype=torch.int32, device=device))
        curr_context_len = torch.tensor([pos_val + 1], dtype=torch.int32, device=device)

        slot_mapping, block_tables_tensor, _ = runner._prepare_inputs(
            seq_ids, curr_context_len, max_blocks = max_blocks
        )

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        logits = runner.execute(
            next_input_id, next_pos, seq_ids, curr_context_len
        )

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        decode_times.append((t_end - t_start) * 1000.0)
        next_input_id.copy_(torch.argmax(logits, dim=-1, keepdim=True))

    avg_itl_ms = sum(decode_times) / len(decode_times)
    tps = 1000.0 / avg_itl_ms

    print(f"Inter-Token Latency    : {avg_itl_ms:.4f} ms/token")
    print(f"Decode Throughput      : {tps:.2f} tokens/sec")


if __name__ == "__main__":
    run_benchmark()
