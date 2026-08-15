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
    input_ids: torch.Tensor,       # [batch_size, seq_len]
    seq_ids: List[int],
    context_lens: torch.Tensor,
    use_torch_sdpa: bool = False   # Flag to switch between custom kernel vs PyTorch SDPA
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
    block_size = model.cache_manager.block_size

    # Precompute slot mappings across all prompt positions once to avoid GPU kernel overhead
    # slot_mappings: [seq_len, batch_size]
    slot_mappings = torch.empty((seq_len, batch_size), dtype=torch.int32, device=input_ids.device)
    for p in range(seq_len):
        for i, seq_id in enumerate(seq_ids):
            b_idx = model.cache_manager.block_tables[seq_id][p // block_size]
            b_off = p % block_size
            slot_mappings[p, i] = b_idx * block_size + b_off

    # Initial Embedding Lookup
    x = model.weights["model.embed_tokens.weight"][input_ids]  # [batch_size, seq_len, hidden_size]

    pos_range = torch.arange(seq_len, dtype=torch.int32, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)

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
        my_cuda_engine_cpp.rope_inplace(q, k, model.cos_table, model.sin_table, pos_range)

        # ---------------------------------------------------------------------
        # 3. Store K/V into Paged KV Cache Pool
        # ---------------------------------------------------------------------
        key_cache = model.cache_manager.kv_cache[l, 0]
        val_cache = model.cache_manager.kv_cache[l, 1]

        for p in range(seq_len):
            my_cuda_engine_cpp.paged_kv_store(
                k[:, p], v[:, p], key_cache, val_cache, slot_mappings[p]
            )

        # ---------------------------------------------------------------------
        # 4. Prefill Sequence Attention
        # ---------------------------------------------------------------------
        if use_torch_sdpa:
            # PyTorch SDPA Path (Native GQA support via repeat_interleave / enable_gqa=True)
            q_attn = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
            k_attn = k.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]
            v_attn = v.transpose(1, 2)  # [batch_size, num_kv_heads, seq_len, head_dim]

            if num_heads != num_kv_heads:
                num_queries_per_kv = num_heads // num_kv_heads
                k_attn = k_attn.repeat_interleave(num_queries_per_kv, dim=1)
                v_attn = v_attn.repeat_interleave(num_queries_per_kv, dim=1)

            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q_attn, k_attn, v_attn, is_causal=True, scale=scale
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        else:
            # Custom CUDA Kernel Path
            attn_out = my_cuda_engine_cpp.prefill_attention(q, k, v, scale)
            attn_out = attn_out.view(batch_size, seq_len, hidden_size)

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

        gate = torch.matmul(x_norm2, w_gate.T)
        up = torch.matmul(x_norm2, w_up.T)

        gate_up = torch.cat([gate, up], dim=-1)
        mlp_intermediate = my_cuda_engine_cpp.swiglu(gate_up)

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
    print("🚀 Starting End-to-End Latency Benchmark for Llama-3.2-3B...")

    config_dict = {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5
    }
    config = Llama3BConfig(config_dict)

    std_hidden = 1.0 / math.sqrt(config.hidden_size)
    std_inter = 1.0 / math.sqrt(config.intermediate_size)

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

    print("Allocating FP16 Weight Tensors...")
    with torch.inference_mode():
        embed_weight = torch.randn((config.vocab_size, config.hidden_size), dtype=torch.float16, device=device) * std_hidden
        weights = {
            "model.embed_tokens.weight": embed_weight,
            "model.norm.weight": torch.ones((config.hidden_size,), dtype=torch.float16, device=device),
            "lm_head.weight": embed_weight
        }

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

    gc.collect()
    torch.cuda.empty_cache()

    model = Llama3BEngineModel(config, weights, cache_manager)
    runner = CUDAGraphRunner(model, max_batch_size=1)

    prompt_len = 512
    generate_tokens = 128
    batch_size = 1
    seq_ids = [0]
    cache_manager.allocate_sequence(0, prompt_len + generate_tokens)

    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), dtype=torch.int64, device=device)
    context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)

    # --- 1. Measure Time to First Token (TTFT) with CUDA Events ---
    with torch.inference_mode():
        # Warmup passes
        for _ in range(3):
            _ = prefill_step(model, prompt_ids, seq_ids, context_lens)
        torch.cuda.synchronize()

        num_prefill_runs = 10
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_prefill_runs):
            _ = prefill_step(model, prompt_ids, seq_ids, context_lens)
        end_event.record()
        torch.cuda.synchronize()

        ttft_ms = start_event.elapsed_time(end_event) / num_prefill_runs

    print(f"\n--- Benchmark Results ---")
    print(f"Prompt Length          : {prompt_len} tokens")
    print(f"Time To First Token    : {ttft_ms:.2f} ms")

    # --- 2. Measure Inter-Token Latency (ITL) with CUDA Events ---
    max_seq_len = prompt_len + generate_tokens
    max_blocks = (max_seq_len + block_size - 1) // block_size

    next_input_id = torch.tensor([[101]], dtype=torch.int64, device=device)
    next_pos = torch.zeros((1, 1), dtype=torch.int32, device=device)
    curr_context_len = torch.zeros((1,), dtype=torch.int32, device=device)

    # Graph capture warmup
    next_pos.fill_(prompt_len)
    curr_context_len.fill_(prompt_len + 1)
    runner._prepare_inputs(seq_ids, curr_context_len, max_blocks=max_blocks)
    runner.capture_graph(next_input_id, next_pos, seq_ids, curr_context_len)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(generate_tokens)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(generate_tokens)]

    print(f"Generating {generate_tokens} tokens...")
    with torch.inference_mode():
        for i in range(generate_tokens):
            pos_val = prompt_len + i
            next_pos.fill_(pos_val)
            curr_context_len.fill_(pos_val + 1)

            runner._prepare_inputs(seq_ids, curr_context_len, max_blocks=max_blocks)

            start_events[i].record()
            logits = runner.execute(next_input_id, next_pos, seq_ids, curr_context_len)
            end_events[i].record()

            next_input_id.copy_(torch.argmax(logits, dim=-1, keepdim=True))

        torch.cuda.synchronize()

    decode_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_itl_ms = sum(decode_times) / len(decode_times)
    tps = 1000.0 / avg_itl_ms

    print(f"Inter-Token Latency    : {avg_itl_ms:.4f} ms/token")
    print(f"Decode Throughput      : {tps:.2f} tokens/sec")


if __name__ == "__main__":
    run_benchmark()
