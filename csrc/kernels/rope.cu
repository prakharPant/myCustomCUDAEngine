#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void rope_inplace_kernel_fp16(
    __half* __restrict__ q,             // [batch_size, seq_len, num_heads, head_dim]
    __half* __restrict__ k,             // [batch_size, seq_len, num_kv_heads, head_dim] (Optional, can be nullptr)
    const float* __restrict__ cos_table,// [max_seq_len, head_dim / 2]
    const float* __restrict__ sin_table,// [max_seq_len, head_dim / 2]
    const int* __restrict__ positions,  // [batch_size, seq_len]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int q_stride_b, int q_stride_s, int q_stride_h,
    int k_stride_b, int k_stride_s, int k_stride_h
) {
    int token_idx = blockIdx.x; // Batch * seq_len element
    int head_idx = blockIdx.y;  // Head index
    int half_dim = head_dim / 2;

    int tid = threadIdx.x;
    if (tid >= half_dim) return;

    // Resolve batch and sequence position
    int pos = positions[token_idx];
    
    // Pointers to cos/sin for this specific token position
    const float* cos_ptr = cos_table + pos * half_dim;
    const float* sin_ptr = sin_table + pos * half_dim;

    float cos_val = cos_ptr[tid];
    float sin_val = sin_ptr[tid];

    // --- 1. Process Query Tensor (Q) ---
    if (head_idx < num_heads) {
        __half* q_head = q + token_idx * q_stride_s + head_idx * q_stride_h;
        
        float q0 = __half2float(q_head[tid]);
        float q1 = __half2float(q_head[tid + half_dim]);

        float q0_rot = q0 * cos_val - q1 * sin_val;
        float q1_rot = q0 * sin_val + q1 * cos_val;

        q_head[tid]            = __float2half(q0_rot);
        q_head[tid + half_dim] = __float2half(q1_rot);
    }

    // --- 2. Process Key Tensor (K) ---
    if (k != nullptr && head_idx < num_kv_heads) {
        __half* k_head = k + token_idx * k_stride_s + head_idx * k_stride_h;

        float k0 = __half2float(k_head[tid]);
        float k1 = __half2float(k_head[tid + half_dim]);

        float k0_rot = k0 * cos_val - k1 * sin_val;
        float k1_rot = k0 * sin_val + k1 * cos_val;

        k_head[tid]            = __float2half(k0_rot);
        k_head[tid + half_dim] = __float2half(k1_rot);
    }
}

void rope_inplace_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor positions
) {
    TORCH_CHECK(query.is_cuda() && cos_table.is_cuda() && sin_table.is_cuda());
    
    int batch_size = query.size(0);
    int seq_len = query.size(1);
    int num_heads = query.size(2);
    int head_dim = query.size(3);
    
    int num_kv_heads = key.defined() ? key.size(2) : 0;

    int total_tokens = batch_size * seq_len;
    int max_heads = std::max(num_heads, num_kv_heads);

    dim3 grid(total_tokens, max_heads);
    dim3 block(std::min(head_dim / 2, 256));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rope_inplace_kernel_fp16<<<grid, block, 0, stream>>>(
        reinterpret_cast<__half*>(query.data_ptr<at::Half>()),
        key.defined() ? reinterpret_cast<__half*>(key.data_ptr<at::Half>()) : nullptr,
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        positions.data_ptr<int>(),
        num_heads,
        num_kv_heads,
        head_dim,
        query.stride(0), query.stride(1), query.stride(2),
        key.defined() ? key.stride(0) : 0, key.defined() ? key.stride(1) : 0, key.defined() ? key.stride(2) : 0
    );
}
