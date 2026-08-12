#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ void ldmatrix_x4(uint32_t* R, const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
        : "r"(smem_addr)
    );
}

// Non-transposed ldmatrix for Matrix V loading
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
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1])
    );
}

// Helper: get the (row, col) that a thread owns for an m16n8 fragment
__device__ __forceinline__ void mma_m16n8_coords(int lane, int& r0, int& r1, int& c0, int& c1) {
    int group = lane % 4;               // 0..3  → which pair of columns
    int row_in_quad = (lane / 4) % 8;   // 0..7
    r0 = row_in_quad;                   // first 8 rows
    r1 = row_in_quad + 8;               // second 8 rows
    c0 = group * 2;
    c1 = c0 + 1;
}

__global__ void level2_tile_attn_kernel(
    const half* __restrict__ Q,  // [16, 16]
    const half* __restrict__ K,  // [16, 16]
    const half* __restrict__ V,  // [16, 16]
    half* __restrict__ O,        // [16, 16]
    const float scale
) {
    const int tid     = threadIdx.x;
    const int lane_id = tid % 32;

    __shared__ half smem_Q[16 * 16];
    __shared__ half smem_K[16 * 16];
    __shared__ half smem_V[16 * 16];
    __shared__ half smem_P[16 * 16];

    // 1. Cooperative load
    for (int i = tid; i < 16 * 16; i += 32) {
        smem_Q[i] = Q[i];
        smem_K[i] = K[i];
        smem_V[i] = V[i];
    }
    __syncthreads();

    // 2. Load Q [16×16]  →  m16n8k16 needs two 8-col tiles
    uint32_t Q_frag[4];
    {
        int q_row = lane_id % 16;
        int q_col = (lane_id / 16) * 8;
        ldmatrix_x4(Q_frag, &smem_Q[q_row * 16 + q_col]);
    }

    // 3. S = Q @ K^T   (two m16n8k16 → full 16×16)
    uint32_t K0_frag[2], K1_frag[2];
    {
        int k0_row = lane_id % 8;
        int k0_col = (lane_id / 8) * 8;
        ldmatrix_x2(K0_frag, &smem_K[k0_row * 16 + k0_col]);

        int k1_row = 8 + (lane_id % 8);
        int k1_col = (lane_id / 8) * 8;
        ldmatrix_x2(K1_frag, &smem_K[k1_row * 16 + k1_col]);
    }

    float S0[4] = {0.f, 0.f, 0.f, 0.f};   // columns 0-7
    float S1[4] = {0.f, 0.f, 0.f, 0.f};   // columns 8-15
    mma_m16n8k16_f32(S0, Q_frag, K0_frag);
    mma_m16n8k16_f32(S1, Q_frag, K1_frag);

    // 4. Scale + causal mask  (now with correct coordinates)
    int r0, r1, c0, c1;
    mma_m16n8_coords(lane_id, r0, r1, c0, c1);

    S0[0] *= scale; S0[1] *= scale; S0[2] *= scale; S0[3] *= scale;
    S1[0] *= scale; S1[1] *= scale; S1[2] *= scale; S1[3] *= scale;

    // causal: set to -inf if column > row
    if (c0 > r0) S0[0] = -1e9f;
    if (c1 > r0) S0[1] = -1e9f;
    if (c0 > r1) S0[2] = -1e9f;
    if (c1 > r1) S0[3] = -1e9f;

    if (c0 + 8 > r0) S1[0] = -1e9f;
    if (c1 + 8 > r0) S1[1] = -1e9f;
    if (c0 + 8 > r1) S1[2] = -1e9f;
    if (c1 + 8 > r1) S1[3] = -1e9f;

    // 5. Online softmax – full-warp reduction for each of the two rows
    float max_r0 = fmaxf(fmaxf(S0[0], S0[1]), fmaxf(S1[0], S1[1]));
    float max_r1 = fmaxf(fmaxf(S0[2], S0[3]), fmaxf(S1[2], S1[3]));

    // also reduce inside the 4-thread group
    max_r0 = fmaxf(max_r0, __shfl_xor_sync(0xffffffff, max_r0, 1));
    max_r0 = fmaxf(max_r0, __shfl_xor_sync(0xffffffff, max_r0, 2));
    max_r1 = fmaxf(max_r1, __shfl_xor_sync(0xffffffff, max_r1, 1));
    max_r1 = fmaxf(max_r1, __shfl_xor_sync(0xffffffff, max_r1, 2));

    float p0_0 = expf(S0[0] - max_r0);
    float p0_1 = expf(S0[1] - max_r0);
    float p1_0 = expf(S1[0] - max_r0);
    float p1_1 = expf(S1[1] - max_r0);

    float p0_2 = expf(S0[2] - max_r1);
    float p0_3 = expf(S0[3] - max_r1);
    float p1_2 = expf(S1[2] - max_r1);
    float p1_3 = expf(S1[3] - max_r1);

    float sum_r0 = p0_0 + p0_1 + p1_0 + p1_1;
    float sum_r1 = p0_2 + p0_3 + p1_2 + p1_3;

    sum_r0 += __shfl_xor_sync(0xffffffff, sum_r0, 1);
    sum_r0 += __shfl_xor_sync(0xffffffff, sum_r0, 2);
    sum_r1 += __shfl_xor_sync(0xffffffff, sum_r1, 1);
    sum_r1 += __shfl_xor_sync(0xffffffff, sum_r1, 2);

    // 6. Write P into shared memory with the *correct* coordinates
    smem_P[r0 * 16 + c0    ] = __float2half(p0_0);
    smem_P[r0 * 16 + c1    ] = __float2half(p0_1);
    smem_P[r0 * 16 + c0 + 8] = __float2half(p1_0);
    smem_P[r0 * 16 + c1 + 8] = __float2half(p1_1);

    smem_P[r1 * 16 + c0    ] = __float2half(p0_2);
    smem_P[r1 * 16 + c1    ] = __float2half(p0_3);
    smem_P[r1 * 16 + c0 + 8] = __float2half(p1_2);
    smem_P[r1 * 16 + c1 + 8] = __float2half(p1_3);
    __syncwarp();

    // 7. O = P @ V
    uint32_t P_frag[4];
    {
        int p_row = lane_id % 16;
        int p_col = (lane_id / 16) * 8;
        ldmatrix_x4(P_frag, &smem_P[p_row * 16 + p_col]);
    }

    uint32_t V0_frag[2], V1_frag[2];
    {
        int v0_row = lane_id % 16; 
        int v0_col = 0;
        int v1_row = lane_id % 16; 
        int v1_col = 8;
        ldmatrix_x2_trans(V0_frag, &smem_V[v0_row * 16 + v0_col]); 
        ldmatrix_x2_trans(V1_frag, &smem_V[v1_row * 16 + v1_col]); 
    }

    float O0[4] = {0.f, 0.f, 0.f, 0.f};
    float O1[4] = {0.f, 0.f, 0.f, 0.f};
    mma_m16n8k16_f32(O0, P_frag, V0_frag);
    mma_m16n8k16_f32(O1, P_frag, V1_frag);

    // 8. Normalize + store (same coordinate helper)
    O0[0] /= sum_r0; O0[1] /= sum_r0;
    O0[2] /= sum_r1; O0[3] /= sum_r1;
    O1[0] /= sum_r0; O1[1] /= sum_r0;
    O1[2] /= sum_r1; O1[3] /= sum_r1;

    O[r0 * 16 + c0    ] = __float2half(O0[0]);
    O[r0 * 16 + c1    ] = __float2half(O0[1]);
    O[r1 * 16 + c0    ] = __float2half(O0[2]);
    O[r1 * 16 + c1    ] = __float2half(O0[3]);

    O[r0 * 16 + c0 + 8] = __float2half(O1[0]);
    O[r0 * 16 + c1 + 8] = __float2half(O1[1]);
    O[r1 * 16 + c0 + 8] = __float2half(O1[2]);
    O[r1 * 16 + c1 + 8] = __float2half(O1[3]);
}

torch::Tensor run_level2_tile_attn(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(Q.device());
    torch::Tensor O = torch::zeros({16, 16}, options);
    level2_tile_attn_kernel<<<1, 32>>>(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        scale
    );
    return O;
}
