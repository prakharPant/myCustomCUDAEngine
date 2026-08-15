import math
import gc
import torch
import torch.nn.functional as F
import my_cuda_engine_cpp
from typing import Tuple
from engine.models.llama import Llama3BConfig


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Compute variance in float32 for stability, then cast back
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def pytorch_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    pos_range: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = q.dtype
    # cos and sin are float32, we cast them back to FP16 just for the rotation math
    cos_pos = cos[pos_range].unsqueeze(2).to(orig_dtype)
    sin_pos = sin[pos_range].unsqueeze(2).to(orig_dtype)

    def apply_rotary(x: torch.Tensor):
        d_half = x.shape[-1] // 2
        x1 = x[..., :d_half]
        x2 = x[..., d_half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        cos_full = torch.cat([cos_pos, cos_pos], dim=-1)
        sin_full = torch.cat([sin_pos, sin_pos], dim=-1)
        return (x * cos_full + rotated * sin_full).to(orig_dtype)

    return apply_rotary(q), apply_rotary(k)


def pytorch_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def time_op(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def run_benchmark():
    device = "cuda"
    prompt_len = 16384
    batch_size = 1
    num_layers = 28

    config_dict = {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5
    }
    config = Llama3BConfig(config_dict)

    std_hidden = 1.0 / math.sqrt(config.hidden_size)
    std_inter = 1.0 / math.sqrt(config.intermediate_size)
    scale = 1.0 / math.sqrt(config.head_dim)

    print("=" * 60)
    print(f" 🔬 1. OPERATOR MICROBENCHMARKS (Seq Len: {prompt_len})")
    print("=" * 60)

    # 1. RMSNorm
    x_sample = torch.empty((batch_size * prompt_len, config.hidden_size), dtype=torch.float16, device=device).normal_()
    w_norm = torch.ones((config.hidden_size,), dtype=torch.float16, device=device)
    t_rmsnorm_custom = time_op(lambda: my_cuda_engine_cpp.rmsnorm(x_sample, w_norm, 1e-5), iters=10, warmup=3) * 1000.0
    t_rmsnorm_pt = time_op(lambda: pytorch_rmsnorm(x_sample, w_norm, 1e-5), iters=10, warmup=3) * 1000.0
    print(f"RMSNorm            | Custom: {t_rmsnorm_custom:8.2f} µs | PyTorch: {t_rmsnorm_pt:8.2f} µs | Speedup: {t_rmsnorm_pt/t_rmsnorm_custom:.2f}x")

    # 2. SwiGLU
    gate_up_sample = torch.empty((batch_size * prompt_len, config.intermediate_size * 2), dtype=torch.float16, device=device).normal_()
    t_swiglu_custom = time_op(lambda: my_cuda_engine_cpp.swiglu(gate_up_sample), iters=10, warmup=3) * 1000.0
    t_swiglu_pt = time_op(lambda: pytorch_swiglu(gate_up_sample), iters=10, warmup=3) * 1000.0
    print(f"SwiGLU             | Custom: {t_swiglu_custom:8.2f} µs | PyTorch: {t_swiglu_pt:8.2f} µs | Speedup: {t_swiglu_pt/t_swiglu_custom:.2f}x")

    # 3. Attention
    q_sample = torch.empty((batch_size, prompt_len, config.num_attention_heads, config.head_dim), dtype=torch.float16, device=device).normal_()
    k_sample = torch.empty((batch_size, prompt_len, config.num_key_value_heads, config.head_dim), dtype=torch.float16, device=device).normal_()
    v_sample = torch.empty((batch_size, prompt_len, config.num_key_value_heads, config.head_dim), dtype=torch.float16, device=device).normal_()
    
    q_attn = q_sample.transpose(1, 2)
    k_attn = k_sample.transpose(1, 2)
    v_attn = v_sample.transpose(1, 2)

    t_attn_custom = time_op(lambda: my_cuda_engine_cpp.prefill_attention(q_sample, k_sample, v_sample, scale), iters=5, warmup=2) * 1000.0
    t_attn_pt = time_op(lambda: F.scaled_dot_product_attention(q_attn, k_attn, v_attn, is_causal=True, scale=scale, enable_gqa=True), iters=5, warmup=2) * 1000.0
    print(f"Prefill Attention  | Custom: {t_attn_custom:8.2f} µs | PyTorch (SDPA): {t_attn_pt:8.2f} µs | Speedup: {t_attn_pt/t_attn_custom:.2f}x")

    # Clean up microbenchmark temporary tensors
    del x_sample, w_norm, gate_up_sample, q_sample, k_sample, v_sample, q_attn, k_attn, v_attn
    gc.collect()
    torch.cuda.empty_cache()

    # --- End-to-End Layer Pass Simulation (Memory-Budgeted) ---
    print("\n" + "=" * 60)
    print(f" 🚀 2. FULL MODEL PREFILL LATENCY (28 Layers, Seq Len: {prompt_len})")
    print("=" * 60)

    # Allocate single representative layer weights (~220 MB total)
    with torch.inference_mode():
        norm_w = torch.ones((config.hidden_size,), dtype=torch.float16, device=device)
        w_q = torch.empty((config.num_attention_heads * config.head_dim, config.hidden_size), dtype=torch.float16, device=device).normal_(0.0, std_hidden)
        w_k = torch.empty((config.num_key_value_heads * config.head_dim, config.hidden_size), dtype=torch.float16, device=device).normal_(0.0, std_hidden)
        w_v = torch.empty((config.num_key_value_heads * config.head_dim, config.hidden_size), dtype=torch.float16, device=device).normal_(0.0, std_hidden)
        w_o = torch.empty((config.hidden_size, config.num_attention_heads * config.head_dim), dtype=torch.float16, device=device).normal_(0.0, std_hidden)
        w_gate = torch.empty((config.intermediate_size, config.hidden_size), dtype=torch.float16, device=device).normal_(0.0, std_hidden)
        w_up = torch.empty((config.intermediate_size, config.hidden_size), dtype=torch.float16, device=device).normal_(0.0, std_hidden)
        w_down = torch.empty((config.hidden_size, config.intermediate_size), dtype=torch.float16, device=device).normal_(0.0, std_inter)

        # FIX: RoPE tables MUST be float32 for the C++ kernel
        cos_table = torch.randn((prompt_len, config.head_dim // 2), dtype=torch.float32, device=device)
        sin_table = torch.randn((prompt_len, config.head_dim // 2), dtype=torch.float32, device=device)
        pos_range = torch.arange(prompt_len, dtype=torch.int32, device=device).unsqueeze(0)

        # Single layer test execution functions
        def custom_layer_forward(x):
            for _ in range(num_layers):
                res = x
                x_norm = my_cuda_engine_cpp.rmsnorm(x.view(-1, config.hidden_size), norm_w, 1e-5).view(batch_size, prompt_len, config.hidden_size)
                
                q = torch.matmul(x_norm, w_q.T).view(batch_size, prompt_len, config.num_attention_heads, config.head_dim)
                k = torch.matmul(x_norm, w_k.T).view(batch_size, prompt_len, config.num_key_value_heads, config.head_dim)
                v = torch.matmul(x_norm, w_v.T).view(batch_size, prompt_len, config.num_key_value_heads, config.head_dim)
                
                my_cuda_engine_cpp.rope_inplace(q, k, cos_table, sin_table, pos_range)
                
                attn = my_cuda_engine_cpp.prefill_attention(q, k, v, scale).view(batch_size, prompt_len, config.hidden_size)
                x = res + torch.matmul(attn, w_o.T)

                res = x
                x_norm2 = my_cuda_engine_cpp.rmsnorm(x.view(-1, config.hidden_size), norm_w, 1e-5)
                gate = torch.matmul(x_norm2, w_gate.T)
                up = torch.matmul(x_norm2, w_up.T)
                gate_up = torch.cat([gate, up], dim=-1)
                
                mlp_out = torch.matmul(my_cuda_engine_cpp.swiglu(gate_up), w_down.T).view(batch_size, prompt_len, config.hidden_size)
                x = res + mlp_out
            return x

        def pytorch_layer_forward(x):
            for _ in range(num_layers):
                res = x
                x_norm = pytorch_rmsnorm(x, norm_w, 1e-5)
                
                q = torch.matmul(x_norm, w_q.T).view(batch_size, prompt_len, config.num_attention_heads, config.head_dim)
                k = torch.matmul(x_norm, w_k.T).view(batch_size, prompt_len, config.num_key_value_heads, config.head_dim)
                v = torch.matmul(x_norm, w_v.T).view(batch_size, prompt_len, config.num_key_value_heads, config.head_dim)
                
                q, k = pytorch_rope(q, k, cos_table, sin_table, pos_range)
                
                q_a = q.transpose(1, 2)
                k_a = k.transpose(1, 2)
                v_a = v.transpose(1, 2)
                
                attn = F.scaled_dot_product_attention(q_a, k_a, v_a, is_causal=True, scale=scale, enable_gqa=True)
                attn = attn.transpose(1, 2).contiguous().view(batch_size, prompt_len, config.hidden_size)
                x = res + torch.matmul(attn, w_o.T)

                res = x
                x_norm2 = pytorch_rmsnorm(x, norm_w, 1e-5)
                gate = torch.matmul(x_norm2, w_gate.T)
                up = torch.matmul(x_norm2, w_up.T)
                gate_up = torch.cat([gate, up], dim=-1)
                
                mlp_out = torch.matmul(pytorch_swiglu(gate_up), w_down.T).view(batch_size, prompt_len, config.hidden_size)
                x = res + mlp_out
            return x

        hidden_states = torch.empty((batch_size, prompt_len, config.hidden_size), dtype=torch.float16, device=device).normal_()

        print("Running Custom Engine...")
        ttft_custom = time_op(lambda: custom_layer_forward(hidden_states), iters=3, warmup=1)
        print(f"-> Custom Engine Prefill Latency : {ttft_custom:.2f} ms")

        # Small GC clean cycle to ensure PyTorch run has maximum free memory
        gc.collect()
        torch.cuda.empty_cache()

        print("Running PyTorch Native Engine...")
        ttft_pytorch = time_op(lambda: pytorch_layer_forward(hidden_states), iters=3, warmup=1)
        print(f"-> PyTorch SDPA Prefill Latency  : {ttft_pytorch:.2f} ms")

        print("\n" + "-" * 60)
        print(f"Overall Speedup (PyTorch / Custom): {ttft_pytorch / ttft_custom:.2f}x")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    run_benchmark()
