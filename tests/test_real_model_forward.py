import torch
import my_cuda_engine_cpp
from engine.weight_loader import WeightLoader
from engine.cache import KVCacheManager
from engine.models.llama import Llama3BConfig, Llama3BEngineModel
from engine.runner import CUDAGraphRunner
from huggingface_hub import snapshot_download


def test_real_llama_3b_forward():
    print("Loading real Llama-3.2-3B weights...")
    model_dir = snapshot_download(
        repo_id="unsloth/Llama-3.2-3B-Instruct",
        allow_patterns=["*.json", "*.safetensors"],
        ignore_patterns=["*.bin", "*.pt", "*.gguf"],
    )

    loader = WeightLoader(model_dir=model_dir, device="cuda")
    weights = loader.load_weights()
    config = Llama3BConfig(loader.config)

    cache_manager = KVCacheManager(
        num_blocks=200,
        block_size=16,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    model = Llama3BEngineModel(config, weights, cache_manager)
    runner = CUDAGraphRunner(model, max_batch_size=2)

    # Decode step simulation
    input_ids = torch.tensor([[128000], [128000]], dtype=torch.int64, device="cuda")
    positions = torch.tensor([[0], [0]], dtype=torch.int32, device="cuda")
    seq_ids = [0, 1]
    context_lens = torch.tensor([1, 1], dtype=torch.int32, device="cuda")

    cache_manager.allocate_sequence(0, 1)
    cache_manager.allocate_sequence(1, 1)

    # Prepare inputs using runner helper to match signature requirements
    slot_mapping, block_tables, max_blocks = runner._prepare_inputs(
        seq_ids, context_lens
    )

    logits = model.decode_step(
        input_ids, positions, context_lens, slot_mapping, block_tables, max_blocks
    )

    assert not torch.isnan(logits).any(), "NaN detected in real model logits!"
    print(
        "✅ Real Llama-3.2-3B Forward Pass Passed! Logits min/max:",
        logits.min().item(),
        logits.max().item(),
    )


if __name__ == "__main__":
    test_real_llama_3b_forward()
