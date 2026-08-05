#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Vectorized scatter kernel: write [batch_size, num_heads, head_dim] into paged block memory
__global__ void reshape_and_cache_kernel(
    const __half* __restrict__ key,       // [batch_size, num_heads, head_dim]
    const __half* __restrict__ value,     // [batch_size, num_heads, head_dim]
    __half* __restrict__ key_cache,       // [num_blocks, block_size, num_heads, head_dim]
    __half* __restrict__ value_cache,     // [num_blocks, block_size, num_heads, head_dim]
    const int* __restrict__ slot_mapping, // [batch_size] -> physical flat token index
    int num_heads,
    int head_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int slot_idx = slot_mapping[seq_idx];

    if (slot_idx < 0) return; // Unused slot

    int block_size = 16; // Fixed block size
    int block_idx = slot_idx / block_size;
    int block_offset = slot_idx % block_size;

    // Source offsets
    int src_offset = (seq_idx * num_heads + head_idx) * head_dim;

    // Target offsets inside paged memory
    int tgt_offset = ((block_idx * block_size + block_offset) * num_heads + head_idx) * head_dim;

    // 128-bit Vectorized Copy (8 FP16 values per load)
    const uint4* src_k = reinterpret_cast<const uint4*>(key + src_offset);
    const uint4* src_v = reinterpret_cast<const uint4*>(value + src_offset);
    uint4* tgt_k = reinterpret_cast<uint4*>(key_cache + tgt_offset);
    uint4* tgt_v = reinterpret_cast<uint4*>(value_cache + tgt_offset);

    int vec_size = head_dim / 8;
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        tgt_k[i] = src_k[i]; // 128-bit store instruction
        tgt_v[i] = src_v[i]; // 128-bit store instruction
    }
}

void paged_kv_store_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping
) {
    int batch_size = key.size(0);
    int num_heads = key.size(1);
    int head_dim = key.size(2);

    dim3 grid(batch_size, num_heads);
    dim3 block(std::min(head_dim / 8, 256));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    reshape_and_cache_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(value_cache.data_ptr<at::Half>()),
        slot_mapping.data_ptr<int>(),
        num_heads,
        head_dim
    );
}
