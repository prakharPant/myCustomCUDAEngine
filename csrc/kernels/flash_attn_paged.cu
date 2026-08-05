#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Warp reduction for maximum value
__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

// Warp reduction for sum
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void paged_attention_decode_kernel(
    const __half *__restrict__ query,       // [batch_size, num_heads, head_dim]
    const __half *__restrict__ key_cache,   // [num_blocks, block_size,
                                            // num_kv_heads, head_dim]
    const __half *__restrict__ value_cache, // [num_blocks, block_size,
                                            // num_kv_heads, head_dim]
    const int *__restrict__ block_tables,   // [batch_size, max_blocks_per_seq]
    const int *__restrict__ context_lens,   // [batch_size]
    __half *__restrict__ output,            // [batch_size, num_heads, head_dim]
    int max_blocks_per_seq, int block_size, int num_heads, int num_kv_heads,
    int head_dim, float scale) {
  int seq_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int tid = threadIdx.x;

  int context_len = context_lens[seq_idx];
  if (context_len <= 0)
    return;

  // Load Query into registers for this thread
  // Assuming head_dim = 128 (standard for Llama-3 / Mistral) handled by 32 warp
  // threads
  const __half *q_ptr = query + (seq_idx * num_heads + head_idx) * head_dim;

  // Each thread in warp handles (head_dim / 32) elements
  constexpr int VEC_SIZE = 4; // 4 FP16 elements per thread for head_dim=128
  float q_reg[VEC_SIZE];

#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    int elem_idx = tid * VEC_SIZE + i;
    q_reg[i] = (elem_idx < head_dim) ? __half2float(q_ptr[elem_idx]) : 0.0f;
  }

  // Online Softmax statistics initialization
  // Explicitly reset all state to prevent register bleeding across graph
  // replays
  float m_prev = -10000.0f;
  float d_prev = 0.0f;
  float acc[VEC_SIZE];
#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    acc[i] = 0.0f;
  }

  int num_blocks = (context_len + block_size - 1) / block_size;
  const int *seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

  // Outer loop over physical KV blocks
  for (int b = 0; b < num_blocks; ++b) {
    int physical_block_id = seq_block_table[b];
    int block_len = min(block_size, context_len - b * block_size);

    // Map KV heads (Grouped Query Attention compatible)
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Inner loop over tokens in block
    for (int t = 0; t < block_len; ++t) {
      int token_offset =
          ((physical_block_id * block_size + t) * num_kv_heads + kv_head_idx) *
          head_dim;
      const __half *k_ptr = key_cache + token_offset;
      const __half *v_ptr = value_cache + token_offset;

      // 1. Compute Q * K^T dot product
      float dot = 0.0f;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        int elem_idx = tid * VEC_SIZE + i;
        if (elem_idx < head_dim) {
          dot += q_reg[i] * __half2float(k_ptr[elem_idx]);
        }
      }

      // Warp reduction for dot product
      float score = warpReduceSum(dot) * scale;

      // DEBUG: Print score for the first warp

//      if (tid == 0 && seq_idx == 0 && head_idx == 0) {
//        printf("DEBUG: seq 0 head 0 score: %f\n", score);
//      }

      // Broadcast score from lane 0 across warp
      score = __shfl_sync(0xffffffff, score, 0);

      // 2. Online Softmax update step
      float m_curr = fmaxf(m_prev, score);
      float alpha = expf(m_prev - m_curr);
      float beta = expf(score - m_curr);

      d_prev = d_prev * alpha + beta;

// 3. Rescale previous accumulator & add beta * V
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        int elem_idx = tid * VEC_SIZE + i;
        float v_val =
            (elem_idx < head_dim) ? __half2float(v_ptr[elem_idx]) : 0.0f;
        acc[i] = acc[i] * alpha + beta * v_val;
      }

      m_prev = m_curr;
    }
  }

  // Final normalization by d_prev
  float inv_d = 1.0f / d_prev;
  __half *out_ptr = output + (seq_idx * num_heads + head_idx) * head_dim;

#pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    int elem_idx = tid * VEC_SIZE + i;
    if (elem_idx < head_dim) {
      out_ptr[elem_idx] = __float2half(acc[i] * inv_d);
    }
  }
}

torch::Tensor paged_attention_decode_cuda(
    torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor block_tables, torch::Tensor context_lens,
    int max_blocks_per_seq, int block_size, float scale) {
  int batch_size = query.size(0);
  int num_heads = query.size(1);
  int head_dim = query.size(2);
  int num_kv_heads = key_cache.size(2);

  auto output = torch::empty_like(query);

  dim3 grid(batch_size, num_heads);
  dim3 block(32); // 1 warp per head
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  paged_attention_decode_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half *>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __half *>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const __half *>(value_cache.data_ptr<at::Half>()),
      block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),
      reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
      max_blocks_per_seq, block_size, num_heads, num_kv_heads, head_dim, scale);

  return output;
}
