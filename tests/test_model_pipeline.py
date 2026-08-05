import math
import my_cuda_engine_cpp
import torch
from engine.cache import KVCacheManager
from engine.models.llama import Llama3BConfig, Llama3BEngineModel
from engine.runner import CUDAGraphRunner


def test_llama_model_forward():
    torch.manual_seed(42)
    device = "cuda"

    # Instantiate miniature config matching 3B shape constraints
    config_dict = {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 2,  # Test 2 layers for fast execution
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
    }
    config = Llama3BConfig(config_dict)

    # Scaling factor for synthetic weights to prevent FP16 overflow
    std_hidden = 1.0 / math.sqrt(config.hidden_size)
    std_inter = 1.0 / math.sqrt(config.intermediate_size)

    # Populate scaled FP16 GPU weights
    weights = {
        "model.embed_tokens.weight": torch.randn(
            (config.vocab_size, config.hidden_size), dtype=torch.float16, device=device
        )
        * std_hidden,
        "model.norm.weight": torch.ones(
            (config.hidden_size,), dtype=torch.float16, device=device
        ),
        "lm_head.weight": torch.randn(
            (config.vocab_size, config.hidden_size), dtype=torch.float16, device=device
        )
        * std_hidden,
    }

    for l in range(config.num_hidden_layers):
        p = f"model.layers.{l}"
        weights[f"{p}.input_layernorm.weight"] = torch.ones(
            (config.hidden_size,), dtype=torch.float16, device=device
        )
        weights[f"{p}.post_attention_layernorm.weight"] = torch.ones(
            (config.hidden_size,), dtype=torch.float16, device=device
        )

        weights[f"{p}.self_attn.q_proj.weight"] = (
            torch.randn(
                (config.num_attention_heads * config.head_dim, config.hidden_size),
                dtype=torch.float16,
                device=device,
            )
            * std_hidden
        )
        weights[f"{p}.self_attn.k_proj.weight"] = (
            torch.randn(
                (config.num_key_value_heads * config.head_dim, config.hidden_size),
                dtype=torch.float16,
                device=device,
            )
            * std_hidden
        )
        weights[f"{p}.self_attn.v_proj.weight"] = (
            torch.randn(
                (config.num_key_value_heads * config.head_dim, config.hidden_size),
                dtype=torch.float16,
                device=device,
            )
            * std_hidden
        )
        weights[f"{p}.self_attn.o_proj.weight"] = (
            torch.randn(
                (config.hidden_size, config.num_attention_heads * config.head_dim),
                dtype=torch.float16,
                device=device,
            )
            * std_hidden
        )

        weights[f"{p}.mlp.gate_proj.weight"] = (
            torch.randn(
                (config.intermediate_size, config.hidden_size),
                dtype=torch.float16,
                device=device,
            )
            * std_hidden
        )
        weights[f"{p}.mlp.up_proj.weight"] = (
            torch.randn(
                (config.intermediate_size, config.hidden_size),
                dtype=torch.float16,
                device=device,
            )
            * std_hidden
        )
        weights[f"{p}.mlp.down_proj.weight"] = (
            torch.randn(
                (config.hidden_size, config.intermediate_size),
                dtype=torch.float16,
                device=device,
            )
            * std_inter
        )

    # Initialize KV Cache pool
    cache_manager = KVCacheManager(
        num_blocks=50,
        block_size=16,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float16,
        device=device,
    )

    # Initialize KV cache with valid values
    cache_manager.kv_cache.normal_(mean=0.0, std=std_hidden)

    model = Llama3BEngineModel(config, weights, cache_manager)
    runner = CUDAGraphRunner(model, max_batch_size=2)

    # Simulate batch of 2 requests
    batch_size = 2
    input_ids = torch.tensor([[100], [200]], dtype=torch.int64, device=device)
    positions = torch.tensor([[10], [25]], dtype=torch.int32, device=device)
    seq_ids = [0, 1]
    context_lens = torch.tensor([11, 26], dtype=torch.int32, device=device)

    # Allocate sequences in cache manager
    cache_manager.allocate_sequence(0, 11)
    cache_manager.allocate_sequence(1, 26)

    # Prepare inputs using runner helper to match signature requirements
    slot_mapping, block_tables, max_blocks = runner._prepare_inputs(
        seq_ids, context_lens
    )

    # Execute decode forward step
    logits = model.decode_step(
        input_ids, positions, context_lens, slot_mapping, block_tables, max_blocks
    )

    assert logits.shape == (
        batch_size,
        config.vocab_size,
    ), f"Unexpected logits shape {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN values detected in output logits!"

    print("✅ Model Pipeline Test Passed! Logits shape:", logits.shape)


if __name__ == "__main__":
    test_llama_model_forward()
