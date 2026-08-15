# ⚡ myCustomCUDAEngine: High-Throughput Bare-Metal Inference

A bare-metal, PyTorch-free LLM inference engine engineered in C++17 and CUDA (Ampere SM 86). Designed from first principles to beat native PyTorch and `torch.compile` (SDPA) on consumer hardware (NVIDIA RTX 3070 Ti 8GB) by eliminating Python runtime dispatch, cutting intermediate High-Bandwidth Memory (HBM) round-trips, and using asynchronous tensor core pipelines.

Built specifically to serve **Llama-3.2-3B** entirely out of custom CUDA kernels.

## 🚀 Performance Highlights

On an NVIDIA GeForce RTX 3070 Ti (8GB VRAM) running Llama-3.2-3B (28 Layers, 3072 Hidden Dim):

* **RMSNorm:** **8.68x faster** than PyTorch (512.41 µs vs. 4450.10 µs).
* **SwiGLU Activation:** **7.68x faster** than PyTorch (458.55 µs vs. 3520.51 µs).
* **Paged FlashAttention Decode:** **1.71x faster** than PyTorch SDPA (1.49 ms vs. 2.56 ms).
* **Asynchronous Prefill Attention:** Outperforms PyTorch SDPA up to **32,768 tokens**.
* **End-to-End 28-Layer Model Prefill:** **1.11x whole-model speedup** (4,676.48 ms vs. 5,169.22 ms at 16k context).

👉 **[Read the Full Benchmark & Systems Engineering Report](docs/BENCHMARKS.md)** for a deep dive into the PTX instructions, memory management, and scaling profiles.

## 🧠 Architecture Overview

`myCustomCUDAEngine` bypasses `torch.nn` entirely. It implements a fully custom forward execution graph utilizing:

1. **Custom Fused Kernels:** Vectorized 128-bit memory accesses (`LDG.E.128`), in-register chunking for SwiGLU, and single-pass shared memory reductions.
2. **Asynchronous Tensor Core Pipelines:** Prefill FlashAttention driven by PTX `cp.async` double-buffering and `mma.sync` hardware matrix math.
3. **Paged KV-Cache Manager:** Zero memory fragmentation via demand-paged non-contiguous block allocations for Keys and Values.
4. **CUDA Graph Runner:** Sub-microsecond execution replay, eliminating Python host-to-device dispatch latency.
5. **Zero-Copy Safetensors Loader:** Direct memory-mapped VRAM loading of FP16 Hugging Face weights.

## 📂 Repository Structure

```text
myCustomCUDAEngine/
├── benchmarks/         # HBM & Latency benchmarking scripts vs PyTorch
├── csrc/
│   ├── bindings.cpp    # PyBind11 C++ interface declarations
│   └── kernels/        # Bare-metal CUDA kernels (RMSNorm, RoPE, SwiGLU, FlashAttention)
├── docs/               # Detailed technical reports and roofline analyses
├── engine/             # Cache Manager, Weight Loader, and CUDA Graph Runner
└── tests/              # End-to-end numerical precision validation suite
```
## 🛠️ Quickstart: Build & Run
1. Install Dependencies & Build Engine

Requires CUDA Toolkit and PyTorch.
Bash

git clone [https://github.com/your-username/myCustomCUDAEngine.git](https://github.com/your-username/myCustomCUDAEngine.git)

cd myCustomCUDAEngine

pip install -e .

2. Run Numerical Precision Tests

Bash

PYTHONPATH=. python tests/test_rmsnorm.py

PYTHONPATH=. python tests/test_paged_attention.py

PYTHONPATH=. python tests/test_cuda_graph.py

3. Run Benchmark Suite

Run the full-engine comparison against PyTorch native execution:
Bash

# Operator microbenchmarks + 28-layer end-to-end prefill comparison
PYTHONPATH=. python benchmarks/benchmark_engine_vs_pytorch.py

# Prefill attention scaling across context windows (128 -> 32768 tokens)
PYTHONPATH=. python benchmarks/benchmark_prefill_attn_revised.py

