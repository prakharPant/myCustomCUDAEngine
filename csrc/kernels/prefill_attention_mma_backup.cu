#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>


// Add these macros at the top of your file, below the includes:
// Standard XOR Swizzling to eliminate ldmatrix memory bank conflicts
#define SWIZZLE_128(row, col) ((row) * 128 + ((col) ^ (((row) & 7) * 8)))
#define SWIZZLE_64(row, col)  ((row) * 64  + ((col) ^ (((row) & 7) * 8)))
#define TILE_M 64   
#define TILE_N 64   
#define HEAD_DIM 128
#define WARP_SIZE 32

// ADD THIS HELPER FUNCTION
__device__ __forceinline__ uint32_t __pack_half2(half a, half b) {
    __half2 res = __halves2half2(a, b);
    return *reinterpret_cast<uint32_t*>(&res);
}

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

__device__ __forceinline__ void ldmatrix_x2(uint32_t* R, const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(R[0]), "=r"(R[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* R, const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(R[0]), "=r"(R[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16_f32(
    float* D, const uint32_t* A, const uint32_t* B
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3])
    );
}



__global__ void prefill_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const float scale_log2
) {
    const int b_idx = blockIdx.z;
    const int h_idx = blockIdx.y;
    const int m_idx = blockIdx.x;  
    
    const int gqa_ratio = num_heads / num_kv_heads;
    const int kv_h_idx = h_idx / gqa_ratio;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char smem_raw[];
    half* smem_Q = reinterpret_cast<half*>(smem_raw);                                
    half* smem_K = smem_Q + (TILE_M * HEAD_DIM);                                     
    half* smem_V = smem_K + (2 * TILE_N * HEAD_DIM);                                 
    half* smem_P = smem_V + (2 * TILE_N * HEAD_DIM);                                 

    const int q_offset = ((b_idx * seq_len + m_idx * TILE_M) * num_heads + h_idx) * HEAD_DIM;
    const int kv_base_offset = (b_idx * seq_len * num_kv_heads + kv_h_idx) * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    // Async Load Q Tile (Swizzled)
    for (int i = tid * 8; i < TILE_M * HEAD_DIM; i += blockDim.x * 8) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        if (m_idx * TILE_M + row < seq_len) {
            cp_async_cg_16(&smem_Q[SWIZZLE_128(row, col)], &Q[q_offset + row * num_heads * HEAD_DIM + col]);
        } else {
            *reinterpret_cast<int4*>(&smem_Q[SWIZZLE_128(row, col)]) = make_int4(0, 0, 0, 0);
        }
    }
    cp_async_commit();

    float m_i[2] = {-1e30f, -1e30f};
    float l_i[2] = {0.0f, 0.0f};    
    float O_acc[16][4] = {0.0f};        

    const int max_kv_tile = min((m_idx + 1) * TILE_M, seq_len);
    const int num_kv_steps = (max_kv_tile + TILE_N - 1) / TILE_N;

    // Double-Buffering Prefetch Stage 0 (Swizzled)
    if (num_kv_steps > 0) {
        for (int i = tid * 8; i < TILE_N * HEAD_DIM; i += blockDim.x * 8) {
            int r = i / HEAD_DIM;
            int c = i % HEAD_DIM;
            if (r < seq_len) {
                cp_async_cg_16(&smem_K[SWIZZLE_128(r, c)], &K[kv_base_offset + r * kv_stride + c]);
                cp_async_cg_16(&smem_V[SWIZZLE_128(r, c)], &V[kv_base_offset + r * kv_stride + c]);
            }
        }
        cp_async_commit();
        cp_async_wait<1>(); 
    } else {
        cp_async_wait<0>();  
    }
    __syncthreads();

    int stage = 0;
    
    for (int k_tile = 0; k_tile < num_kv_steps; ++k_tile) {
        int next_stage = stage ^ 1;
        int next_tile = k_tile + 1;

        if (next_tile < num_kv_steps) {
            half* smem_K_next = smem_K + next_stage * (TILE_N * HEAD_DIM);
            half* smem_V_next = smem_V + next_stage * (TILE_N * HEAD_DIM);
            for (int i = tid * 8; i < TILE_N * HEAD_DIM; i += blockDim.x * 8) {
                int r = next_tile * TILE_N + (i / HEAD_DIM);
                int c = i % HEAD_DIM;
                if (r < seq_len) {
                    cp_async_cg_16(&smem_K_next[SWIZZLE_128(i / HEAD_DIM, c)], &K[kv_base_offset + r * kv_stride + c]);
                    cp_async_cg_16(&smem_V_next[SWIZZLE_128(i / HEAD_DIM, c)], &V[kv_base_offset + r * kv_stride + c]);
                }
            }
            cp_async_commit();
            cp_async_wait<1>(); 
        } else {
            cp_async_wait<0>(); 
        }
        __syncthreads();

        half* smem_K_curr = smem_K + stage * (TILE_N * HEAD_DIM);
        half* smem_V_curr = smem_V + stage * (TILE_N * HEAD_DIM);

        float S_acc[8][4] = {0.0f};

        // 1. Compute Q * K^T (Swizzled reads eliminate bank conflicts)
        #pragma unroll
        for (int k_step = 0; k_step < HEAD_DIM; k_step += 16) {
            uint32_t q_frag[4];
            int q_row = warp_id * 16 + (lane_id % 16);
            int q_col = k_step + (lane_id / 16) * 8;
            ldmatrix_x4(q_frag, &smem_Q[SWIZZLE_128(q_row, q_col)]);

            #pragma unroll
            for (int n_sub = 0; n_sub < 8; ++n_sub) {
                uint32_t k_frag[2];
                int k_row = n_sub * 8 + (lane_id % 8);
                int k_col = k_step + ((lane_id / 8) % 2) * 8;
                
                ldmatrix_x2(k_frag, &smem_K_curr[SWIZZLE_128(k_row, k_col)]);
                mma_m16n8k16_f32(S_acc[n_sub], q_frag, k_frag);
            }
        }

        // 2 & 3. Fused Scale, Causal Masking, and Online Softmax Max Reduction
        int row_0 = warp_id * 16 + (lane_id / 4);
        int row_1 = row_0 + 8;
        int g_row_0 = m_idx * TILE_M + row_0;
        int g_row_1 = m_idx * TILE_M + row_1;
        int col_base = k_tile * TILE_N + (lane_id % 4) * 2;
        
        bool is_causal_tile = (m_idx == k_tile);
        float local_max_0 = -1e30f, local_max_1 = -1e30f;

        #pragma unroll
        for (int n_sub = 0; n_sub < 8; ++n_sub) {
            int c0 = col_base + n_sub * 8;
            int c1 = c0 + 1;

            // Fused Scale
            S_acc[n_sub][0] *= scale_log2;
            S_acc[n_sub][1] *= scale_log2;
            S_acc[n_sub][2] *= scale_log2;
            S_acc[n_sub][3] *= scale_log2;

            // Fused Mask
            if (is_causal_tile) {
                if (c0 > g_row_0) S_acc[n_sub][0] = -1e30f;
                if (c1 > g_row_0) S_acc[n_sub][1] = -1e30f;
                if (c0 > g_row_1) S_acc[n_sub][2] = -1e30f;
                if (c1 > g_row_1) S_acc[n_sub][3] = -1e30f;
            }

            // Fused Max (Using fast hardware fmaxf instead of generic max)
            local_max_0 = fmaxf(local_max_0, fmaxf(S_acc[n_sub][0], S_acc[n_sub][1]));
            local_max_1 = fmaxf(local_max_1, fmaxf(S_acc[n_sub][2], S_acc[n_sub][3]));
        }

        #pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
            local_max_0 = fmaxf(local_max_0, __shfl_xor_sync(0xffffffff, local_max_0, mask));
            local_max_1 = fmaxf(local_max_1, __shfl_xor_sync(0xffffffff, local_max_1, mask));
        }

        float m_new_0 = fmaxf(m_i[0], local_max_0);
        float m_new_1 = fmaxf(m_i[1], local_max_1);

        float alpha_0 = exp2f(m_i[0] - m_new_0);
        float alpha_1 = exp2f(m_i[1] - m_new_1);

        #pragma unroll
        for (int v_sub = 0; v_sub < 16; ++v_sub) {
            O_acc[v_sub][0] *= alpha_0;
            O_acc[v_sub][1] *= alpha_0;
            O_acc[v_sub][2] *= alpha_1;
            O_acc[v_sub][3] *= alpha_1;
        }

        float P_sum_0 = 0.0f, P_sum_1 = 0.0f;

        #pragma unroll
        for (int n_sub = 0; n_sub < 8; ++n_sub) {
            float p0 = exp2f(S_acc[n_sub][0] - m_new_0);
            float p1 = exp2f(S_acc[n_sub][1] - m_new_0);
            float p2 = exp2f(S_acc[n_sub][2] - m_new_1);
            float p3 = exp2f(S_acc[n_sub][3] - m_new_1);

            P_sum_0 += (p0 + p1);
            P_sum_1 += (p2 + p3);

            int c0 = n_sub * 8 + (lane_id % 4) * 2;
            
            // Pack and store to Swizzled layout
            uint32_t p_01 = __pack_half2(__float2half(p0), __float2half(p1));
            uint32_t p_23 = __pack_half2(__float2half(p2), __float2half(p3));

            *reinterpret_cast<uint32_t*>(&smem_P[SWIZZLE_64(row_0, c0)]) = p_01;
            *reinterpret_cast<uint32_t*>(&smem_P[SWIZZLE_64(row_1, c0)]) = p_23;
        }

        #pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
            P_sum_0 += __shfl_xor_sync(0xffffffff, P_sum_0, mask);
            P_sum_1 += __shfl_xor_sync(0xffffffff, P_sum_1, mask);
        }

        l_i[0] = l_i[0] * alpha_0 + P_sum_0;
        l_i[1] = l_i[1] * alpha_1 + P_sum_1;
        m_i[0] = m_new_0;
        m_i[1] = m_new_1;

        __syncwarp();   

        // 4. Multiply P * V (Swizzled reads)
        #pragma unroll
        for (int p_step = 0; p_step < 4; ++p_step) {
            uint32_t P_frag[4];
            int p_row = warp_id * 16 + (lane_id % 16);
            int p_col = p_step * 16 + (lane_id / 16) * 8;

            ldmatrix_x4(P_frag, &smem_P[SWIZZLE_64(p_row, p_col)]);

            #pragma unroll
            for (int v_sub = 0; v_sub < 16; ++v_sub) {
                uint32_t v_frag[2];
                int v_row = p_step * 16 + ((lane_id / 8) % 2) * 8 + (lane_id % 8);
                int v_col = v_sub * 8;

                ldmatrix_x2_trans(v_frag, &smem_V_curr[SWIZZLE_128(v_row, v_col)]);
                mma_m16n8k16_f32(O_acc[v_sub], P_frag, v_frag);
            }
        }

        stage = next_stage;
    }

    __syncthreads();  
    half* smem_O = smem_K;  

    int row_0 = warp_id * 16 + (lane_id / 4);
    int row_1 = row_0 + 8;
    int g_row_0 = m_idx * TILE_M + row_0;
    int g_row_1 = m_idx * TILE_M + row_1;

    #pragma unroll
    for (int v_sub = 0; v_sub < 16; ++v_sub) {
        int col_0 = v_sub * 8 + (lane_id % 4) * 2;
        
        float o0 = (l_i[0] > 0.0f) ? (O_acc[v_sub][0] / l_i[0]) : O_acc[v_sub][0];
        float o1 = (l_i[0] > 0.0f) ? (O_acc[v_sub][1] / l_i[0]) : O_acc[v_sub][1];
        float o2 = (l_i[1] > 0.0f) ? (O_acc[v_sub][2] / l_i[1]) : O_acc[v_sub][2];
        float o3 = (l_i[1] > 0.0f) ? (O_acc[v_sub][3] / l_i[1]) : O_acc[v_sub][3];

        uint32_t o_01 = __pack_half2(__float2half(o0), __float2half(o1));
        uint32_t o_23 = __pack_half2(__float2half(o2), __float2half(o3));

        *reinterpret_cast<uint32_t*>(&smem_O[SWIZZLE_128(row_0, col_0)]) = o_01;
        *reinterpret_cast<uint32_t*>(&smem_O[SWIZZLE_128(row_1, col_0)]) = o_23;
    }

    __syncthreads();

    for (int i = tid * 8; i < TILE_M * HEAD_DIM; i += blockDim.x * 8) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        if (m_idx * TILE_M + row < seq_len) {
            int out_off = ((b_idx * seq_len + m_idx * TILE_M + row) * num_heads + h_idx) * HEAD_DIM + col;
            *reinterpret_cast<int4*>(&O[out_off]) = *reinterpret_cast<int4*>(&smem_O[SWIZZLE_128(row, col)]);
        }
    }
}


torch::Tensor prefill_attention_cuda(
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
    dim3 block(128);  

    size_t smem_bytes = (TILE_M * HEAD_DIM + 2 * 2 * TILE_N * HEAD_DIM + TILE_M * TILE_N) * sizeof(half);

    cudaError_t err = cudaFuncSetAttribute(
        prefill_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    TORCH_CHECK(err == cudaSuccess, "Failed to set max dynamic shared memory size: ", cudaGetErrorString(err));

    // Fast log2 translation trick for standard __expf
    const float log2e = 1.4426950408889634f;
    float scale_log2 = scale * log2e;

    prefill_attention_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        seq_len,
        num_heads,
        num_kv_heads,
        scale_log2
    );

    return out;
}
