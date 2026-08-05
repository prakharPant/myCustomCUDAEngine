import math
from typing import Dict, List, Optional

import my_cuda_engine_cpp
import torch

from engine.cache import KVCacheManager


class Llama3BConfig:
    def __init__(self, config_dict: dict):
        self.hidden_size: int = config_dict.get("hidden_size", 3072)
        self.intermediate_size: int = config_dict.get("intermediate_size", 8192)
        self.num_hidden_layers: int = config_dict.get("num_hidden_layers", 28)
        self.num_attention_heads: int = config_dict.get("num_attention_heads", 24)
        self.num_key_value_heads: int = config_dict.get("num_key_value_heads", 8)
        self.head_dim: int = config_dict.get("head_dim", 128)
        self.vocab_size: int = config_dict.get("vocab_size", 128256)
        self.rms_norm_eps: float = config_dict.get("rms_norm_eps", 1e-5)
        self.rope_theta: float = config_dict.get("rope_theta", 500000.0)
        self.max_position_embeddings: int = config_dict.get(
            "max_position_embeddings", 131072
        )


class Llama3BEngineModel:
    def __init__(
        self,
        config: Llama3BConfig,
        weights: Dict[str, torch.Tensor],
        cache_manager: KVCacheManager,
    ):
        self.config = config
        self.weights = weights
        self.cache_manager = cache_manager

        # Scaling factor for query scores
        self.scale = 1.0 / math.sqrt(self.config.head_dim)

        # Precompute RoPE Cos/Sin tables up to max positions on GPU
        self.cos_table, self.sin_table = self._precompute_rope_tables(
            max_seq_len=4096,  # Working context range for test
            head_dim=self.config.head_dim,
            base=self.config.rope_theta,
        )

    def _precompute_rope_tables(self, max_seq_len: int, head_dim: int, base: float):
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device="cuda")
                / head_dim
            )
        )
        t = torch.arange(max_seq_len, dtype=torch.float32, device="cuda")
        freqs = torch.outer(t, inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

    def decode_step(
        self,
        input_ids: torch.Tensor,  # [batch_size, 1]
        positions: torch.Tensor,  # [batch_size, 1]
        context_lens: torch.Tensor,  # [batch_size]
        slot_mapping: torch.Tensor,  # [batch_size] (Pre-calculated on host)
        block_tables_tensor: torch.Tensor,  # [batch_size, max_blocks_per_seq]
        max_blocks_per_seq: int,  # Primitive integer (Passed from runner)
    ) -> torch.Tensor:
        """
        Pure, zero-host-sync forward step for CUDA Graph compatibility.
        """
        batch_size = input_ids.size(0)

        # 1. Embedding lookup
        embed_weight = self.weights["model.embed_tokens.weight"]
        x = embed_weight[input_ids]

        # Iterate through Transformer Layers
        for l in range(self.config.num_hidden_layers):
            layer_prefix = f"model.layers.{l}"

            # --- A. Input RMSNorm ---
            norm_w = self.weights[f"{layer_prefix}.input_layernorm.weight"]
            x_norm = my_cuda_engine_cpp.rmsnorm(
                x.squeeze(1), norm_w, self.config.rms_norm_eps
            )

            # --- B. Projections ---
            w_q = self.weights[f"{layer_prefix}.self_attn.q_proj.weight"]
            w_k = self.weights[f"{layer_prefix}.self_attn.k_proj.weight"]
            w_v = self.weights[f"{layer_prefix}.self_attn.v_proj.weight"]

            # Change lines in decode_step:
            q = (
                torch.matmul(x_norm, w_q.T)
                .view(
                    batch_size, 1, self.config.num_attention_heads, self.config.head_dim
                )
                .contiguous()
            )
            k = (
                torch.matmul(x_norm, w_k.T)
                .view(
                    batch_size, 1, self.config.num_key_value_heads, self.config.head_dim
                )
                .contiguous()
            )
            v = (
                torch.matmul(x_norm, w_v.T)
                .view(
                    batch_size, 1, self.config.num_key_value_heads, self.config.head_dim
                )
                .contiguous()
                .contiguous()
            )

            # --- C. Fused RoPE ---
            my_cuda_engine_cpp.rope_inplace(
                q, k, self.cos_table, self.sin_table, positions
            )

            # --- D. Paged KV Store ---
            key_cache = self.cache_manager.kv_cache[l, 0]
            val_cache = self.cache_manager.kv_cache[l, 1]

            my_cuda_engine_cpp.paged_kv_store(
                k.squeeze(1), v.squeeze(1), key_cache, val_cache, slot_mapping
            )

            # --- E. Paged FlashAttention Decode ---
            attn_out = my_cuda_engine_cpp.paged_attention_decode(
                q.squeeze(1),
                key_cache,
                val_cache,
                block_tables_tensor,
                context_lens,
                max_blocks_per_seq,
                self.cache_manager.block_size,
                self.scale,
            )

            # --- F. Out Projection & Residual ---
            w_o = self.weights[f"{layer_prefix}.self_attn.o_proj.weight"]
            attn_proj = torch.matmul(attn_out.view(batch_size, -1), w_o.T)
            x = x.squeeze(1) + attn_proj

            # --- G. Post-Norm & SwiGLU ---
            post_norm_w = self.weights[
                f"{layer_prefix}.post_attention_layernorm.weight"
            ]
            x_post_norm = my_cuda_engine_cpp.rmsnorm(
                x, post_norm_w, self.config.rms_norm_eps
            )

            w_gate = self.weights[f"{layer_prefix}.mlp.gate_proj.weight"]
            w_up = self.weights[f"{layer_prefix}.mlp.up_proj.weight"]
            w_down = self.weights[f"{layer_prefix}.mlp.down_proj.weight"]

            gate_proj = torch.matmul(x_post_norm, w_gate.T)
            up_proj = torch.matmul(x_post_norm, w_up.T)

            gate_up = torch.cat([gate_proj, up_proj], dim=-1)
            mlp_act = my_cuda_engine_cpp.swiglu(gate_up)

            mlp_out = torch.matmul(mlp_act, w_down.T)
            x = (x + mlp_out).unsqueeze(1)

        # --- Final Head Projection ---
        final_norm_w = self.weights["model.norm.weight"]
        x_final = my_cuda_engine_cpp.rmsnorm(
            x.squeeze(1), final_norm_w, self.config.rms_norm_eps
        )

        lm_head_w = self.weights.get("lm_head.weight", embed_weight)
        return torch.matmul(x_final, lm_head_w.T)
