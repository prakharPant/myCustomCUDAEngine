import os
import torch
from huggingface_hub import snapshot_download
from engine.weight_loader import WeightLoader

def test_load_llama_3b():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    print(f"Downloading/Locating model repository: {model_id}...")
    # Note: Requires HF approval or token if repository is gated
    try:
        model_dir = snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.safetensors"],
            ignore_patterns=["*.bin", "*.pt", "*.gguf"]
        )
    except Exception as e:
        print(f"Standard HF download failed ({e}). Falling back to uncensored/community mirror if needed...")
        model_dir = snapshot_download(
            repo_id="unsloth/Llama-3.2-3B-Instruct",
            allow_patterns=["*.json", "*.safetensors"],
            ignore_patterns=["*.bin", "*.pt", "*.gguf"]
        )

    # Instantiate Loader
    loader = WeightLoader(model_dir=model_dir, device="cuda")
    weights = loader.load_weights()

    # Verify Llama 3.2 3B Layer Count & Dimensions
    num_layers = loader.config.get("num_hidden_layers", 28)
    hidden_size = loader.config.get("hidden_size", 3072)
    intermediate_size = loader.config.get("intermediate_size", 8192)

    # Inspect Specific Key Weights
    embed_weight = weights["model.embed_tokens.weight"]
    norm_weight = weights["model.norm.weight"]
    gate_proj_l0 = weights["model.layers.0.mlp.gate_proj.weight"]

    assert embed_weight.shape[1] == hidden_size, f"Expected hidden size {hidden_size}, got {embed_weight.shape[1]}"
    assert norm_weight.shape[0] == hidden_size, f"Expected norm size {hidden_size}, got {norm_weight.shape[0]}"
    assert gate_proj_l0.shape[0] == intermediate_size, f"Expected MLP intermediate size {intermediate_size}, got {gate_proj_l0.shape[0]}"

    # Print VRAM Usage Report
    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    
    print(f"\n--- VRAM Footprint Report (RTX 3070 Ti) ---")
    print(f"Allocated Weight Memory : {allocated_gb:.2f} GB")
    print(f"Reserved CUDA Memory    : {reserved_gb:.2f} GB")
    print("✅ Llama-3.2-3B Weight Loader Test Passed!")

if __name__ == "__main__":
    test_load_llama_3b()
