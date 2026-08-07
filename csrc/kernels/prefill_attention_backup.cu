#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

using namespace nvcuda;

constexpr int BR = 64;       // Query tile size
constexpr int BC = 64;       // Key/Value tile size
constexpr int HEAD_DIM = 128;

// Fully Corrected FlashAttention-2 CUDA Kernel
__global__ void prefill_attention_fa2_wmma_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ output,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_tile_idx = blockIdx.z * BR;

    int tid = threadIdx.x;            // 128 threads (4 warps)
    int warp_id = tid / 32;           // 0..3

    int warp_r_offset = (warp_id / 2) * 32; // 0 or 32 (Row group covers 32 rows)
    int warp_c_offset = (warp_id % 2) * 32; // 0 or 32 (Col group covers 64 cols in O/V, 32 cols in S/P)
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // -------------------------------------------------------------------------
    // Dynamic Shared Memory Layout (Exact 48.25 KB Allocation)
    // -------------------------------------------------------------------------
    extern __shared__ char smem_raw[];

    // Offset 0 (16 KB): Persistent Q Tile [64, 128] fp16
    __half* s_q = reinterpret_cast<__half*>(smem_raw);
    
    // Offset 16384 (16 KB): Persistent V Tile [64, 128] fp16
    __half* s_v = s_q + BR * HEAD_DIM;

    // Offset 32768 (16 KB): Aliased Transient Buffer
    // - Step A: Used as s_k [64, 128] fp16 (16 KB)
    // - Step B: Used as s_s [64, 64] float (16 KB)
    // - Step B/C: Used as s_p [64, 64] fp16 (8 KB)
    __half* s_k = s_v + BC * HEAD_DIM;
    float*  s_s = reinterpret_cast<float*>(s_k);
    __half* s_p = reinterpret_cast<__half*>(s_k);

    // Offset 49152 (256 B): Fixed placement for s_alpha after the 16 KB transient buffer
    float* s_alpha = reinterpret_cast<float*>(smem_raw + 3 * BR * HEAD_DIM * sizeof(__half));

    // Online Softmax State per Row
    float m_i = -1e20f;
    float l_i = 0.0f;

    // Warp Accumulator Fragments: [2][4] -> 2 row frags x 4 col frags = 32 rows x 64 cols per warp
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[2][4];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            wmma::fill_fragment(o_frag[r][c], 0.0f);
        }
    }

    // 1. Cooperative Load Q Tile into Shared Memory
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        int load_vec_idx = tid + step * 128;
        int r = load_vec_idx / 16;
        int c_vec = load_vec_idx % 16;
        int global_q_pos = q_tile_idx + r;

        uint4 zero_u4 = make_uint4(0, 0, 0, 0);
        if (r < BR && global_q_pos < seq_len) {
            int q_offset = ((batch_idx * seq_len + global_q_pos) * num_heads + head_idx) * head_dim + c_vec * 8;
            reinterpret_cast<uint4*>(s_q + r * HEAD_DIM)[c_vec] = reinterpret_cast<const uint4*>(q + q_offset)[0];
        } else if (r < BR) {
            reinterpret_cast<uint4*>(s_q + r * HEAD_DIM)[c_vec] = zero_u4;
        }
    }
    __syncthreads();

    int max_k_tiles = (q_tile_idx + BR + BC - 1) / BC;

    // 2. Loop over Causal K/V Tiles
    for (int k_tile = 0; k_tile < max_k_tiles; ++k_tile) {
        int k_tile_start = k_tile * BC;

        __syncthreads();
        #pragma unroll
        for (int step = 0; step < 8; ++step) {
            int load_vec_idx = tid + step * 128;
            int r = load_vec_idx / 16;
            int c_vec = load_vec_idx % 16;
            int global_k_pos = k_tile_start + r;

            uint4 zero_u4 = make_uint4(0, 0, 0, 0);
            if (r < BC && global_k_pos < seq_len) {
                int kv_offset = ((batch_idx * seq_len + global_k_pos) * num_kv_heads + kv_head_idx) * head_dim + c_vec * 8;
                reinterpret_cast<uint4*>(s_k + r * HEAD_DIM)[c_vec] = reinterpret_cast<const uint4*>(k + kv_offset)[0];
                reinterpret_cast<uint4*>(s_v + r * HEAD_DIM)[c_vec] = reinterpret_cast<const uint4*>(v + kv_offset)[0];
            } else if (r < BC) {
                reinterpret_cast<uint4*>(s_k + r * HEAD_DIM)[c_vec] = zero_u4;
                reinterpret_cast<uint4*>(s_v + r * HEAD_DIM)[c_vec] = zero_u4;
            }
        }
        __syncthreads();

        // STEP A: GEMM 1 -> S = Q * K^T [64, 128] x [128, 64] -> [64, 64]
        // S is 64 columns wide -> Each warp handles 32 cols (2 col fragments)
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[2][2];

        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                wmma::fill_fragment(s_frag[r][c], 0.0f);
            }
        }

        #pragma unroll
        for (int k_step = 0; k_step < HEAD_DIM; k_step += 16) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                wmma::load_matrix_sync(q_frag, s_q + (warp_r_offset + r * 16) * HEAD_DIM + k_step, HEAD_DIM);
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    wmma::load_matrix_sync(k_frag, s_k + (warp_c_offset + c * 16) * HEAD_DIM + k_step, HEAD_DIM);
                    wmma::mma_sync(s_frag[r][c], q_frag, k_frag, s_frag[r][c]);
                }
            }
        }

        // Store unscaled scores S to shared memory (s_s)
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                wmma::store_matrix_sync(
                    s_s + (warp_r_offset + r * 16) * BC + (warp_c_offset + c * 16),
                    s_frag[r][c], BC, wmma::mem_row_major
                );
            }
        }
        __syncthreads(); // s_k is dead; s_s contains unscaled scores

        // STEP B: Online Softmax Reduction & Store P into s_p
        if (tid < BR) {
            int row_idx = tid;
            int global_q = q_tile_idx + row_idx;

            float row_max = -1e20f;
            #pragma unroll
            for (int col = 0; col < BC; ++col) {
                int global_k = k_tile_start + col;
                float score = s_s[row_idx * BC + col] * scale;
                if (global_k > global_q || global_q >= seq_len) {
                    score = -1e20f;
                }
                s_s[row_idx * BC + col] = score;
                row_max = fmaxf(row_max, score);
            }

            float m_new = fmaxf(m_i, row_max);
            float alpha = expf(m_i - m_new);

            float row_sum = 0.0f;
            #pragma unroll
            for (int col = 0; col < BC; ++col) {
                float p_val = expf(s_s[row_idx * BC + col] - m_new);
                s_p[row_idx * BC + col] = __float2half(p_val);
                row_sum += p_val;
            }

            m_i = m_new;
            l_i = l_i * alpha + row_sum;
            s_alpha[row_idx] = alpha;
        }
        __syncthreads();

        // In-Register Rescaling of Accumulator Fragments (across all 4 col fragments)
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            int row_in_tile = warp_r_offset + r * 16;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                #pragma unroll
                for (int i = 0; i < o_frag[r][c].num_elements; ++i) {
                    int sub_row = i / 2;
                    float alpha_val = s_alpha[row_in_tile + sub_row];
                    o_frag[r][c].x[i] *= alpha_val;
                }
            }
        }

        // STEP C: GEMM 2 -> O = O + P * V [64, 64] x [64, 128] -> [64, 128]
        // O is 128 columns wide -> Each warp handles 64 cols (4 col fragments: c = 0..3)
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;

        #pragma unroll
        for (int k_step = 0; k_step < BC; k_step += 16) {
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                wmma::load_matrix_sync(p_frag, s_p + (warp_r_offset + r * 16) * BC + k_step, BC);
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    int v_col_offset = warp_c_offset * 2 + c * 16; // Covers 0, 16, 32, 48 (warps 0/2) and 64, 80, 96, 112 (warps 1/3)
                    wmma::load_matrix_sync(v_frag, s_v + k_step * HEAD_DIM + v_col_offset, HEAD_DIM);
                    wmma::mma_sync(o_frag[r][c], p_frag, v_frag, o_frag[r][c]);
                }
            }
        }
        __syncthreads();
    }

    // ---------------------------------------------------------------------------
    // STEP D: Normalize Output by l_i and Write to Global Memory
    // ---------------------------------------------------------------------------
    // Safely reuse the start of smem_raw as a 32 KB [64, 128] float staging buffer.
    float* s_out_stage = reinterpret_cast<float*>(smem_raw);

    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int out_col_offset = warp_c_offset * 2 + c * 16;
            wmma::store_matrix_sync(
                s_out_stage + (warp_r_offset + r * 16) * HEAD_DIM + out_col_offset,
                o_frag[r][c], HEAD_DIM, wmma::mem_row_major
            );
        }
    }
    __syncthreads();

    if (tid < BR) {
        int row_idx = tid;
        int global_q = q_tile_idx + row_idx;

        if (global_q < seq_len) {
            float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            int out_offset = ((batch_idx * seq_len + global_q) * num_heads + head_idx) * head_dim;
            uint4* out_u4_ptr = reinterpret_cast<uint4*>(output + out_offset);

            #pragma unroll
            for (int c_vec = 0; c_vec < 16; ++c_vec) {
                uint4 out_u4;
                __half2* h2_ptr = reinterpret_cast<__half2*>(&out_u4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float val0 = s_out_stage[row_idx * HEAD_DIM + c_vec * 8 + i * 2] * inv_l;
                    float val1 = s_out_stage[row_idx * HEAD_DIM + c_vec * 8 + i * 2 + 1] * inv_l;
                    h2_ptr[i] = __floats2half2_rn(val0, val1);
                }
                out_u4_ptr[c_vec] = out_u4;
            }
        }
    }
}

torch::Tensor prefill_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    float scale
) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda());

    int batch_size = query.size(0);
    int seq_len = query.size(1);
    int num_heads = query.size(2);
    int head_dim = query.size(3);
    int num_kv_heads = key.size(2);

    auto output = torch::empty_like(query);

    dim3 block(128); // 4 warps
    dim3 grid(batch_size, num_heads, (seq_len + BR - 1) / BR);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Exact allocation: 16 KB (Q) + 16 KB (V) + 16 KB (Transient K/S/P) + 256 B (Alpha) = 48.25 KB
    size_t smem_size = (3 * BR * HEAD_DIM * sizeof(__half)) + (BR * sizeof(float));

    cudaFuncSetAttribute(
        prefill_attention_fa2_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    prefill_attention_fa2_wmma_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        seq_len, num_heads, num_kv_heads, head_dim, scale
    );

    return output;
}
