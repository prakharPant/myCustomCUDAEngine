import os
import json
import glob
import gc
import torch
from typing import Dict, Any
from safetensors.torch import load_file

class WeightLoader:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {self.model_dir}")
        with open(config_path, "r") as f:
            return json.load(f)

    def load_weights(self) -> Dict[str, torch.Tensor]:
        safetensor_files = sorted(glob.glob(os.path.join(self.model_dir, "*.safetensors")))
        
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {self.model_dir}")

        weights: Dict[str, torch.Tensor] = {}
        print(f"📦 Loading weights from {len(safetensor_files)} safetensors shard(s)...")

        for file_path in safetensor_files:
            print(f"   -> Reading shard to CPU: {os.path.basename(file_path)}")
            # Load shard to CPU first to avoid VRAM allocation spikes
            shard_weights = load_file(file_path, device="cpu")
            
            for key, tensor in shard_weights.items():
                # Cast to FP16 directly onto CUDA device
                weights[key] = tensor.to(dtype=torch.float16, device=self.device)

            # Free CPU shard memory
            del shard_weights
            gc.collect()
            torch.cuda.empty_cache()

        print(f"✅ Loaded {len(weights)} weight tensors into GPU memory.")
        return weights
