#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_M 64  // Block Q rows
#define TILE_N 64  // Block KV rows
#define HEAD_DIM 128
#define WARP_SIZE 32

// Inline PTX wrappers for Hopper/Ampere hardware primitives
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t* R, const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16_f32(
    float* D, const uint32_t* A, const uint32_t* B
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]),
          "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3])
    );
}

// FlashAttention-2 Prefill Kernel with GQA, cp.async, and MMA.sync
__global__ void prefill_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const float scale
) {
    const int b_idx = blockIdx.z;
    const int h_idx = blockIdx.y;
    const int m_idx = blockIdx.x; // Block tile along Q sequence length
    
    const int gqa_ratio = num_heads / num_kv_heads;
    const int kv_h_idx = h_idx / gqa_ratio;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Allocate Shared Memory for Q, Double-Buffered K, and V
    extern __shared__ char smem_raw[];
    half* smem_Q = reinterpret_cast<half*>(smem_raw);                                   // Size: TILE_M * HEAD_DIM
    half* smem_K = smem_Q + (TILE_M * HEAD_DIM);                                        // Size: 2 * TILE_N * HEAD_DIM
    half* smem_V = smem_K + (2 * TILE_N * HEAD_DIM);                                    // Size: 2 * TILE_N * HEAD_DIM

    const int q_offset = ((b_idx * seq_len + m_idx * TILE_M) * num_heads + h_idx) * HEAD_DIM;
    const int kv_base_offset = (b_idx * seq_len * num_kv_heads + kv_h_idx) * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    // 1. Load Q Tile into Shared Memory
    for (int i = tid * 8; i < TILE_M * HEAD_DIM; i += blockDim.x * 8) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        if (m_idx * TILE_M + row < seq_len) {
            *reinterpret_cast<int4*>(&smem_Q[row * HEAD_DIM + col]) = 
                *reinterpret_cast<const int4*>(&Q[q_offset + row * num_heads * HEAD_DIM + col]);
        } else {
            *reinterpret_cast<int4*>(&smem_Q[row * HEAD_DIM + col]) = make_int4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // FlashAttention Online Softmax State Buffers (Registers)
    float m_i[2] = {-1e30f, -1e30f}; // Row max trackers for warp thread
    float l_i[2] = {0.0f, 0.0f};     // Normalization sums for warp thread
    float O_acc[8] = {0.0f};         // Accumulator for output GEMM

    const int max_kv_tile = min((m_idx + 1) * TILE_M, seq_len); // Causal mask bound
    const int num_kv_steps = (max_kv_tile + TILE_N - 1) / TILE_N;

    // 2. Prefetch first K/V tile into Stage 0 using cp.async
    if (num_kv_steps > 0) {
        int kv_row = 0;
        for (int i = tid * 8; i < TILE_N * HEAD_DIM; i += blockDim.x * 8) {
            int r = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            if (r < seq_len) {
                cp_async_cg_16(&smem_K[r * HEAD_DIM + c], &K[kv_base_offset + r * kv_stride + c]);
                cp_async_cg_16(&smem_V[r * HEAD_DIM + c], &V[kv_base_offset + r * kv_stride + c]);
            }
        }
        cp_async_commit();
    }

    int stage = 0;
    
    // 3. Main Pipelined Loop over K/V Tiles
    for (int k_tile = 0; k_tile < num_kv_steps; ++k_tile) {
        int next_stage = stage ^ 1;
        int next_tile = k_tile + 1;

        // Issue cp.async for Next Stage (Stage k+1)
        if (next_tile < num_kv_steps) {
            half* smem_K_next = smem_K + next_stage * (TILE_N * HEAD_DIM);
            half* smem_V_next = smem_V + next_stage * (TILE_N * HEAD_DIM);
            for (int i = tid * 8; i < TILE_N * HEAD_DIM; i += blockDim.x * 8) {
                int r = next_tile * TILE_N + (i / HEAD_DIM);
                int c = i % HEAD_DIM;
                if (r < seq_len) {
                    cp_async_cg_16(&smem_K_next[(i / HEAD_DIM) * HEAD_DIM + c], &K[kv_base_offset + r * kv_stride + c]);
                    cp_async_cg_16(&smem_V_next[(i / HEAD_DIM) * HEAD_DIM + c], &V[kv_base_offset + r * kv_stride + c]);
                }
            }
            cp_async_commit();
        }

        // Wait for Current Stage (Stage k)
        cp_async_wait<1>();
        __syncthreads();

        half* smem_K_curr = smem_K + stage * (TILE_N * HEAD_DIM);
        half* smem_V_curr = smem_V + stage * (TILE_N * HEAD_DIM);

        // Compute S_tile = Q_tile * K_tile^T using ldmatrix + mma.sync
        float S_acc[4] = {0.0f};
        uint32_t q_frag[4], k_frag[4];

        ldmatrix_x4(q_frag, &smem_Q[(warp_id * 16 + (lane_id % 16)) * HEAD_DIM + (lane_id / 16) * 16]);
        ldmatrix_x4(k_frag, &smem_K_curr[(lane_id % 8) * HEAD_DIM]);

        mma_m16n8k16_f32(S_acc, q_frag, k_frag);

        // Compute Online Softmax Updates
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            S_acc[i] *= scale;
        }

        // Causal Masking
        int row_idx = m_idx * TILE_M + warp_id * 16 + (lane_id / 4);
        int col_idx = k_tile * TILE_N + (lane_id % 4) * 2;
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (col_idx + i > row_idx) {
                S_acc[i] = -1e30f;
                S_acc[i + 2] = -1e30f;
            }
        }

        // Update m_i and rescale previous O_acc
        float m_curr = max(S_acc[0], S_acc[1]);
        float m_prev = m_i[0];
        float m_new = max(m_prev, m_curr);
        
        float alpha = __expf(m_prev - m_new);
        float p0 = __expf(S_acc[0] - m_new);
        float p1 = __expf(S_acc[1] - m_new);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            O_acc[i] *= alpha;
        }

        l_i[0] = l_i[0] * alpha + p0 + p1;
        m_i[0] = m_new;

        // Multiply P_tile * V_tile into O_acc
        uint32_t p_frag[2], v_frag[4];
        __half2 packed =
            __halves2half2(__float2half(p0), __float2half(p1));
        p_frag[0] = reinterpret_cast<const uint32_t*>(&packed)[0];
        p_frag[1] = p_frag[0];

        ldmatrix_x4(v_frag, &smem_V_curr[(lane_id % 8) * HEAD_DIM]);
        mma_m16n8k16_f32(O_acc, p_frag, v_frag);

        stage = next_stage;
    }

    cp_async_wait<0>();
    __syncthreads();

    // 4. Final Scale Normalization and Output Store
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (l_i[0] > 0.0f) {
            O_acc[i] /= l_i[0];
        }
    }

    int out_offset = ((b_idx * seq_len + m_idx * TILE_M + warp_id * 16 + (lane_id / 4)) * num_heads + h_idx) * HEAD_DIM;
    if (m_idx * TILE_M + warp_id * 16 + (lane_id / 4) < seq_len) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            O[out_offset + (lane_id % 4) * 2 + i] = __float2half(O_acc[i]);
        }
    }
}

// C++ Entry Point for PyTorch
torch::Tensor prefill_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    float scale
) {
    const int batch_size = q.size(0);
    const int seq_len = q.size(1);
    const int num_heads = q.size(2);
    const int num_kv_heads = k.size(2);
    const int head_dim = q.size(3);

    auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    torch::Tensor out = torch::empty({batch_size, seq_len, num_heads, head_dim}, options);

    dim3 grid((seq_len + TILE_M - 1) / TILE_M, num_heads, batch_size);
    dim3 block(128); // 4 Warps

    size_t smem_bytes = (TILE_M * HEAD_DIM + 2 * 2 * TILE_N * HEAD_DIM) * sizeof(half);

    prefill_attention_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        seq_len,
        num_heads,
        num_kv_heads,
        scale
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prefill_attention", &prefill_attention, "Prefill Attention Kernel (CUDA)");
}
