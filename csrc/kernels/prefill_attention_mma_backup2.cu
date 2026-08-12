#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// FlashAttention-2 optimized Tile Sizes
#define TILE_M 128    
#define TILE_N 64    
#define HEAD_DIM 128
#define WARP_SIZE 32

// Bitwise Swizzle Macros to prevent Shared Memory Bank Conflicts
#define SWIZZLE_128(row, col) (((row) << 7) | ((col) ^ (((row) & 7) << 3)))
#define SWIZZLE_64(row, col)  (((row) << 6) | ((col) ^ (((row) & 7) << 3)))

__device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
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

__device__ __forceinline__ float fast_exp2(float x) {
    float res;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

__device__ __forceinline__ uint32_t pack_f32_to_h2(float a, float b) {
    __half2 res = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&res);
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
    const int warp_id = tid >> 5;  
    const int lane_id = tid & 31;   

    // TILE_M = 128. 8 Warps split exactly across M (16 rows per warp)
    const int warp_m = warp_id; 

    extern __shared__ half smem_raw[];
    half* smem_Q = smem_raw;                                   // 32 KB
    half* smem_K[2] = {smem_Q + 16384, smem_Q + 16384 + 8192}; // 16 KB + 16 KB
    half* smem_V[2] = {smem_K[1] + 8192, smem_K[1] + 16384};   // 16 KB + 16 KB
    half* smem_P = smem_Q;                                     // Alias Q for P later

    const int q_offset = ((b_idx * seq_len + m_idx * TILE_M) * num_heads + h_idx) * HEAD_DIM;
    const int kv_base_offset = (b_idx * seq_len * num_kv_heads + kv_h_idx) * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    // 1. Load Q into Shared Memory
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = tid + i * 256; 
        int row = idx / 16;
        int col = (idx % 16) * 8;
        if (m_idx * TILE_M + row < seq_len) {
            cp_async_cg_16(&smem_Q[SWIZZLE_128(row, col)], &Q[q_offset + row * num_heads * HEAD_DIM + col]);
        } else {
            *reinterpret_cast<int4*>(&smem_Q[SWIZZLE_128(row, col)]) = make_int4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // 2. Load Q into Registers (FlashAttention-2 trick)
    uint32_t q_frags[8][4];
    #pragma unroll
    for (int k_step = 0; k_step < 8; ++k_step) {
        int q_row = warp_m * 16 + (lane_id % 16);
        int q_col = k_step * 16 + (lane_id / 16) * 8; // Perfectly 16-Byte Aligned
        ldmatrix_x4(q_frags[k_step], &smem_Q[SWIZZLE_128(q_row, q_col)]);
    }
    __syncthreads(); // Q is in registers, smem_Q is now free

    float m_i[2] = {-1e30f, -1e30f};
    float l_i[2] = {0.0f, 0.0f};    
    float O_acc[16][4] = {0.0f};        

    const int max_kv_tile = min((m_idx + 1) * TILE_M, seq_len);
    const int num_kv_steps = (max_kv_tile + TILE_N - 1) / TILE_N;

    auto load_kv = [&](int k_tile_idx, int stage_idx) {
        int r_base = k_tile_idx * TILE_N;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 256;
            int row = idx / 16;
            int col = (idx % 16) * 8;
            if (r_base + row < seq_len) {
                cp_async_ca_16(&smem_K[stage_idx][SWIZZLE_128(row, col)], &K[kv_base_offset + (r_base + row) * kv_stride + col]);
                cp_async_ca_16(&smem_V[stage_idx][SWIZZLE_128(row, col)], &V[kv_base_offset + (r_base + row) * kv_stride + col]);
            }
        }
    };

    // Prologue: Issue first stage
    if (0 < num_kv_steps) {
        load_kv(0, 0);
        cp_async_commit();
    }

    for (int k_tile = 0; k_tile < num_kv_steps; ++k_tile) {
        int stage = k_tile % 2;
        int next_stage = (k_tile + 1) % 2;
        
        if (k_tile + 1 < num_kv_steps) {
            load_kv(k_tile + 1, next_stage);
            cp_async_commit();
        }

        if (k_tile + 1 < num_kv_steps) cp_async_wait<1>();
        else cp_async_wait<0>();
        __syncthreads();

        float S_acc[8][4] = {0.0f}; // 8 steps of 8 -> 64 cols

        // Q * K^T
        #pragma unroll
        for (int k_step = 0; k_step < 8; ++k_step) {
            #pragma unroll
            for (int n_step = 0; n_step < 8; ++n_step) {
                uint32_t k_frag[2];
                // [FIXED] Proper addressing for an 8x16 ldmatrix.x2 read 
                int k_row = n_step * 8 + (lane_id % 8);
                int k_col = k_step * 16 + (lane_id / 8) * 8;
                ldmatrix_x2(k_frag, &smem_K[stage][SWIZZLE_128(k_row, k_col)]);
                mma_m16n8k16_f32(S_acc[n_step], q_frags[k_step], k_frag);
            }
        }

        // Fused Scale, Causal Masking, Softmax
        float local_max[2] = {-1e30f, -1e30f};
        #pragma unroll
        for (int n_step = 0; n_step < 8; ++n_step) {
            int c0 = k_tile * TILE_N + n_step * 8 + (lane_id % 4) * 2;
            int c1 = c0 + 1;
            int r0 = m_idx * TILE_M + warp_m * 16 + (lane_id / 4);
            int r1 = r0 + 8;
            
            S_acc[n_step][0] *= scale_log2; S_acc[n_step][1] *= scale_log2;
            S_acc[n_step][2] *= scale_log2; S_acc[n_step][3] *= scale_log2;

            if (c0 > r0) S_acc[n_step][0] = -1e30f;
            if (c1 > r0) S_acc[n_step][1] = -1e30f;
            if (c0 > r1) S_acc[n_step][2] = -1e30f;
            if (c1 > r1) S_acc[n_step][3] = -1e30f;
            
            local_max[0] = fmaxf(local_max[0], fmaxf(S_acc[n_step][0], S_acc[n_step][1]));
            local_max[1] = fmaxf(local_max[1], fmaxf(S_acc[n_step][2], S_acc[n_step][3]));
        }

        #pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) { // Reduce local max horizontally across warp columns
            local_max[0] = fmaxf(local_max[0], __shfl_xor_sync(0xffffffff, local_max[0], mask));
            local_max[1] = fmaxf(local_max[1], __shfl_xor_sync(0xffffffff, local_max[1], mask));
        }

        float m_new[2] = { fmaxf(m_i[0], local_max[0]), fmaxf(m_i[1], local_max[1]) };
        float alpha[2] = { fast_exp2(m_i[0] - m_new[0]), fast_exp2(m_i[1] - m_new[1]) };

        #pragma unroll
        for (int v_step = 0; v_step < 16; ++v_step) {
            O_acc[v_step][0] *= alpha[0]; O_acc[v_step][1] *= alpha[0];
            O_acc[v_step][2] *= alpha[1]; O_acc[v_step][3] *= alpha[1];
        }

        float P_sum[2] = {0.0f, 0.0f};
        
        #pragma unroll
        for (int n_step = 0; n_step < 8; ++n_step) {
            float p0 = fast_exp2(S_acc[n_step][0] - m_new[0]);
            float p1 = fast_exp2(S_acc[n_step][1] - m_new[0]);
            float p2 = fast_exp2(S_acc[n_step][2] - m_new[1]);
            float p3 = fast_exp2(S_acc[n_step][3] - m_new[1]);
            
            P_sum[0] += (p0 + p1);
            P_sum[1] += (p2 + p3);
            
            int p_row0 = warp_m * 16 + (lane_id / 4);
            int p_row1 = p_row0 + 8;
            int p_col0 = n_step * 8 + (lane_id % 4) * 2;
            
            // Write to smem_P to reset layout for matrix multiplication
            *reinterpret_cast<half2*>(&smem_P[SWIZZLE_64(p_row0, p_col0)]) = __floats2half2_rn(p0, p1);
            *reinterpret_cast<half2*>(&smem_P[SWIZZLE_64(p_row1, p_col0)]) = __floats2half2_rn(p2, p3);
        }

        #pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
            P_sum[0] += __shfl_xor_sync(0xffffffff, P_sum[0], mask);
            P_sum[1] += __shfl_xor_sync(0xffffffff, P_sum[1], mask);
        }

        l_i[0] = l_i[0] * alpha[0] + P_sum[0];
        l_i[1] = l_i[1] * alpha[1] + P_sum[1];
        m_i[0] = m_new[0];
        m_i[1] = m_new[1];

        __syncthreads(); 

        // P * V
        #pragma unroll
        for (int k_step = 0; k_step < 4; ++k_step) {
            uint32_t p_frag[4];
            int p_row = warp_m * 16 + (lane_id % 16);
            int p_col = k_step * 16 + (lane_id / 16) * 8;
            ldmatrix_x4(p_frag, &smem_P[SWIZZLE_64(p_row, p_col)]);
            
            #pragma unroll
            for (int v_step = 0; v_step < 16; ++v_step) {
                uint32_t v_frag[2];
                // [FIXED] Proper addressing for a 16x8 ldmatrix.x2.trans read (reads 2x vertically stacked 8x8 blocks)
                int v_row = k_step * 16 + (lane_id % 16);
                int v_col = v_step * 8; 
                ldmatrix_x2_trans(v_frag, &smem_V[stage][SWIZZLE_128(v_row, v_col)]);
                mma_m16n8k16_f32(O_acc[v_step], p_frag, v_frag);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int v_step = 0; v_step < 16; ++v_step) {
        int col_0 = v_step * 8 + (lane_id % 4) * 2;
        int row_0 = warp_m * 16 + (lane_id / 4);
        int row_1 = row_0 + 8;
        
        float o0 = (l_i[0] > 0.0f) ? (O_acc[v_step][0] / l_i[0]) : O_acc[v_step][0];
        float o1 = (l_i[0] > 0.0f) ? (O_acc[v_step][1] / l_i[0]) : O_acc[v_step][1];
        float o2 = (l_i[1] > 0.0f) ? (O_acc[v_step][2] / l_i[1]) : O_acc[v_step][2];
        float o3 = (l_i[1] > 0.0f) ? (O_acc[v_step][3] / l_i[1]) : O_acc[v_step][3];

        *reinterpret_cast<uint32_t*>(&smem_P[SWIZZLE_128(row_0, col_0)]) = pack_f32_to_h2(o0, o1);
        *reinterpret_cast<uint32_t*>(&smem_P[SWIZZLE_128(row_1, col_0)]) = pack_f32_to_h2(o2, o3);
    }
    
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int idx = tid + i * 256; 
        int row = idx / 16;
        int col = (idx % 16) * 8;
        if (m_idx * TILE_M + row < seq_len) {
            int out_off = ((b_idx * seq_len + m_idx * TILE_M + row) * num_heads + h_idx) * HEAD_DIM + col;
            *reinterpret_cast<int4*>(&O[out_off]) = *reinterpret_cast<int4*>(&smem_P[SWIZZLE_128(row, col)]);
        }
    }
}

torch::Tensor prefill_attention_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale
) {
    const int batch_size = q.size(0);
    const int seq_len = q.size(1);
    const int num_heads = q.size(2);
    const int num_kv_heads = k.size(2);
    const int head_dim = q.size(3);

    auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
    torch::Tensor out = torch::empty({batch_size, seq_len, num_heads, head_dim}, options);

    dim3 grid((seq_len + TILE_M - 1) / TILE_M, num_heads, batch_size);
    dim3 block(256);

    // 96 KB SMEM Allocation -> Fits strictly under sm_86's 100KB limit
    size_t smem_bytes = 96 * 1024;

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            prefill_attention_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
        attr_set = true;
    }

    const float log2e = 1.4426950408889634f;
    float scale_log2 = scale * log2e;

    prefill_attention_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        seq_len, num_heads, num_kv_heads, scale_log2
    );

    return out;
}
