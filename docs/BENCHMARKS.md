# 🔬 Systems Engineering & Benchmark Report

This document outlines the architectural decisions, PTX-level kernel optimizations, and exhaustive benchmark data that enable `myCustomCUDAEngine` to outperform native PyTorch on consumer hardware.

## 🖥️ Hardware Target & Model Spec
* **GPU:** NVIDIA GeForce RTX 3070 Ti (8 GB GDDR6X, Ampere Architecture, SM 86).
* **Compute Constraints:** Memory-bandwidth bound for vector ops; Compute bound for large GEMMs.
* **Target Model:** Llama-3.2-3B (28 Layers, 3072 Hidden Dim, 8192 Intermediate Dim, 24 Query Heads, 8 KV Heads, FP16 precision).

---

## 📊 1. End-to-End Latency & Operator Microbenchmarks

At a massive context length of **16,384 tokens**, custom fused kernels fundamentally bypass PyTorch's VRAM allocation overheads.

| Operator / Layer | PyTorch Baseline | Custom CUDA Engine | Speedup | Constraint Addressed |
| :--- | :--- | :--- | :--- | :--- |
| **RMSNorm** ($N=16k$) | 4,450.10 µs | **512.41 µs** | **8.68x** | HBM Bandwidth / Memory Roundtrips |
| **SwiGLU** ($N=16k$) | 3,520.51 µs | **458.55 µs** | **7.68x** | VRAM Intermediates / VRAM Allocation |
| **Paged Decode Attn** | 2.5588 ms | **1.4958 ms** | **1.71x** | Memory Fragmentation & Indexing |
| **Full Prefill (28L)** | 5,169.22 ms | **4,676.48 ms** | **1.11x** | End-to-End Execution Overhead |

---

## 📈 2. Asynchronous Prefill Attention Scaling Profile

Our custom FlashAttention kernel (leveraging PTX asynchronous copies) successfully matches and outpaces PyTorch's native Scaled Dot Product Attention (SDPA), which is backed by cuDNN/FlashAttention-2.

| Sequence Length | Custom CUDA Engine (ms) | PyTorch SDPA (ms) | Speedup | Max Numerical $\Delta$ |
| :--- | :--- | :--- | :--- | :--- |
| **128** | 0.0195 | 0.0186 | 0.95x | $< 1.95 \times 10^{-3}$ |
| **2,048** | 1.0284 | 0.9427 | 0.92x | $< 1.95 \times 10^{-3}$ |
| **4,096** | 3.6389 | 3.5560 | 0.98x | $< 1.95 \times 10^{-3}$ |
| **8,192** | 14.1706 | 13.9628 | 0.99x | $< 1.95 \times 10^{-3}$ |
| **16,384** | **60.1235** | **61.6674** | **1.03x** | $< 1.95 \times 10^{-3}$ |
| **32,768** | **270.3895** | **278.2944** | **1.03x** | $< 1.95 \times 10^{-3}$ |

---

## 🛠️ 3. Deep Kernel Analysis & PTX Optimizations

### Asynchronous Tensor Core Pipeline (`prefill_attention.cu`)
To achieve peak FLOPS during attention prefill, the engine overlaps memory fetches with matrix math:
* **`cp.async.cg.shared.global`:** Global memory transfers for the next key/value tile bypass the L1/Register file entirely, streaming directly into shared memory.
* **`ldmatrix.m8n8.x4`:** Shared memory data is distributed directly into thread registers natively formatted for Tensor Core consumption.
* **`mma.sync`:** Matrix products are executed on Ampere Hardware Tensor Cores in mixed FP16/FP32 precision.
* **Warp-Level Max-Stabilization:** By tracking the running maximum of the softmax denominator, `__any_sync` warp votes are used to skip 64+ FP32 multiplication instructions per loop if the maximum has stabilized (which occurs >90% of the time).

### 128-Bit Vectorized Fused Normalization (`rmsnorm.cu`)
* **Vectorization:** Reads 8 FP16 elements per thread using `uint4` loads (`LDG.E.128`), saturating the GDDR6X memory bus.
* **Single-Pass Reduction:** Computes variance using intra-warp `__shfl_down_sync` and shared memory reductions, never flushing intermediate tensors to global memory.

### In-Register SwiGLU Activation (`swiglu.cu`)
* Traditional PyTorch executes $\text{SiLU}(\text{gate}) \times \text{up}$ by slicing the tensor, writing the sliced tensors to VRAM, computing SiLU, writing to VRAM, and multiplying.
* Our kernel fuses the slice, activation, and multiplication into a single in-register pass, eliminating two full $[B, S, 8192]$ HBM allocations.

### Paged KV-Cache & Zero-Overhead Runner
* **Block Allocator:** Pre-allocates a pool of 16-token physical blocks, defeating the memory fragmentation that causes Out-Of-Memory (OOM) errors on 8GB GPUs.
* **CUDA Graphs:** The entire decode step is captured into a static memory execution graph. Replaying the graph via hardware triggers drops host dispatch latency from ~1.5 ms down to sub-microsecond levels.
