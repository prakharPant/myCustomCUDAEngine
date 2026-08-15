#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_M 64     
#define TILE_N 32      
#define HEAD_DIM 128
#define WARP_SIZE 32

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

__device__ __forceinline__ float fast_rcp(float x) {
    float res;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(res) : "f"(x));
    return res;
}

__device__ __forceinline__ uint32_t pack_f32_to_h2(float a, float b) {
    uint32_t ha = __half_as_ushort(__float2half_rn(a));
    uint32_t hb = __half_as_ushort(__float2half_rn(b));
    return (hb << 16) | ha;
}

#define LDMATRIX_X4(r, smem_ptr) \
    do { \
        uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr); \
        asm volatile( \
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
            : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) \
            : "r"(smem_addr) \
        ); \
    } while(0)

#define LDMATRIX_X2(r0, r1, smem_ptr) \
    do { \
        uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr); \
        asm volatile( \
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" \
            : "=r"(r0), "=r"(r1) \
            : "r"(smem_addr) \
        ); \
    } while(0)

#define LDMATRIX_X2_TRANS(r0, r1, smem_ptr) \
    do { \
        uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr); \
        asm volatile( \
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" \
            : "=r"(r0), "=r"(r1) \
            : "r"(smem_addr) \
        ); \
    } while(0)

#define MMA_M16N8K16_F32(d, a, b0, b1) \
    asm volatile( \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n" \
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w) \
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), \
          "r"(b0), "r"(b1) \
    )

#define LOAD_KV_TILE(k_tile_idx, K_PTR, V_PTR) \
    do { \
        int r_base = (k_tile_idx) * TILE_N; \
        bool is_valid_kv = (r_base + TILE_N <= seq_len); \
        _Pragma("unroll 4") \
        for (int i = 0; i < 4; ++i) { \
            int idx = tid + i * 128; \
            int row = idx / 16; \
            int col = (idx % 16) * 8; \
            if (is_valid_kv || r_base + row < seq_len) { \
                cp_async_cg_16(&(K_PTR)[SWIZZLE_128(row, col)], &K[kv_base_offset + (r_base + row) * kv_stride + col]); \
                cp_async_cg_16(&(V_PTR)[SWIZZLE_128(row, col)], &V[kv_base_offset + (r_base + row) * kv_stride + col]); \
            } \
        } \
    } while (0)

// Pre-scales Q using half-precision math, removing 64 FP32 mul operations per loop
#define PRE_SCALE_Q_FRAG(Q_FRAG) \
    do { \
        half2* q0 = (half2*)&(Q_FRAG).x; *q0 = __hmul2(*q0, scale_h2); \
        half2* q1 = (half2*)&(Q_FRAG).y; *q1 = __hmul2(*q1, scale_h2); \
        half2* q2 = (half2*)&(Q_FRAG).z; *q2 = __hmul2(*q2, scale_h2); \
        half2* q3 = (half2*)&(Q_FRAG).w; *q3 = __hmul2(*q3, scale_h2); \
    } while(0)

#define ZERO_FLOAT4(var) var.x = 0.0f; var.y = 0.0f; var.z = 0.0f; var.w = 0.0f;

#define UPDATE_MAX(S_ACC_INST) \
    do { \
        local_max_0 = fmaxf(local_max_0, fmaxf(S_ACC_INST.x, S_ACC_INST.y)); \
        local_max_1 = fmaxf(local_max_1, fmaxf(S_ACC_INST.z, S_ACC_INST.w)); \
    } while (0)

#define MASK_S_ACC(N_STEP, S_ACC_INST) \
    do { \
        int c0 = k_tile * TILE_N + (N_STEP) * 8 + (lane_id % 4) * 2; \
        int c1 = c0 + 1; \
        if (c0 > r0) S_ACC_INST.x = -1e30f; \
        if (c1 > r0) S_ACC_INST.y = -1e30f; \
        if (c0 > r1) S_ACC_INST.z = -1e30f; \
        if (c1 > r1) S_ACC_INST.w = -1e30f; \
    } while(0)

// Uses warp aggregation to skip FP32 math if alpha == 1 (which it will be >90% of the time)
#define SCALE_O_ACC(O_ACC_INST) \
    do { \
        if (need_scale_0) { O_ACC_INST.x *= alpha_0; O_ACC_INST.y *= alpha_0; } \
        if (need_scale_1) { O_ACC_INST.z *= alpha_1; O_ACC_INST.w *= alpha_1; } \
    } while(0)

#define COMPUTE_P_AND_STORE(N_STEP, S_ACC_INST) \
    do { \
        float p0 = (S_ACC_INST.x < -1e20f) ? 0.0f : fast_exp2(S_ACC_INST.x - m_new_0); \
        float p1 = (S_ACC_INST.y < -1e20f) ? 0.0f : fast_exp2(S_ACC_INST.y - m_new_0); \
        float p2 = (S_ACC_INST.z < -1e20f) ? 0.0f : fast_exp2(S_ACC_INST.z - m_new_1); \
        float p3 = (S_ACC_INST.w < -1e20f) ? 0.0f : fast_exp2(S_ACC_INST.w - m_new_1); \
        P_sum_0 += (p0 + p1); \
        P_sum_1 += (p2 + p3); \
        int p_col0 = (N_STEP) * 8 + (lane_id % 4) * 2; \
        *reinterpret_cast<uint32_t*>(&smem_P[SWIZZLE_64(p_row0, p_col0)]) = pack_f32_to_h2(p0, p1); \
        *reinterpret_cast<uint32_t*>(&smem_P[SWIZZLE_64(p_row1, p_col0)]) = pack_f32_to_h2(p2, p3); \
    } while(0)

#define COMPUTE_S_ACC_STEP(K_STEP, Q_FRAG) \
    do { \
        uint32_t k0_0, k0_1, k1_0, k1_1, k2_0, k2_1, k3_0, k3_1; \
        int k_col = (K_STEP) * 16 + (lane_id / 8) * 8; \
        LDMATRIX_X2(k0_0, k0_1, &compute_K[SWIZZLE_128(0 * 8 + (lane_id % 8), k_col)]); \
        LDMATRIX_X2(k1_0, k1_1, &compute_K[SWIZZLE_128(1 * 8 + (lane_id % 8), k_col)]); \
        LDMATRIX_X2(k2_0, k2_1, &compute_K[SWIZZLE_128(2 * 8 + (lane_id % 8), k_col)]); \
        LDMATRIX_X2(k3_0, k3_1, &compute_K[SWIZZLE_128(3 * 8 + (lane_id % 8), k_col)]); \
        MMA_M16N8K16_F32(S_acc_0, Q_FRAG, k0_0, k0_1); \
        MMA_M16N8K16_F32(S_acc_1, Q_FRAG, k1_0, k1_1); \
        MMA_M16N8K16_F32(S_acc_2, Q_FRAG, k2_0, k2_1); \
        MMA_M16N8K16_F32(S_acc_3, Q_FRAG, k3_0, k3_1); \
    } while(0)

#define COMPUTE_O_ACC_GROUP(V_START, O_ACC_0, O_ACC_1, O_ACC_2, O_ACC_3) \
    do { \
        uint32_t v0_0, v0_1, v1_0, v1_1, v2_0, v2_1, v3_0, v3_1; \
        LDMATRIX_X2_TRANS(v0_0, v0_1, &compute_V[SWIZZLE_128(v_row, (V_START + 0) * 8)]); \
        LDMATRIX_X2_TRANS(v1_0, v1_1, &compute_V[SWIZZLE_128(v_row, (V_START + 1) * 8)]); \
        LDMATRIX_X2_TRANS(v2_0, v2_1, &compute_V[SWIZZLE_128(v_row, (V_START + 2) * 8)]); \
        LDMATRIX_X2_TRANS(v3_0, v3_1, &compute_V[SWIZZLE_128(v_row, (V_START + 3) * 8)]); \
        MMA_M16N8K16_F32(O_ACC_0, p_frag, v0_0, v0_1); \
        MMA_M16N8K16_F32(O_ACC_1, p_frag, v1_0, v1_1); \
        MMA_M16N8K16_F32(O_ACC_2, p_frag, v2_0, v2_1); \
        MMA_M16N8K16_F32(O_ACC_3, p_frag, v3_0, v3_1); \
    } while(0)

#define STORE_O_ACC(V_STEP, O_ACC_INST) \
    do { \
        int col_0 = (V_STEP) * 8 + (lane_id % 4) * 2; \
        float o0 = O_ACC_INST.x * rcp_l0; \
        float o1 = O_ACC_INST.y * rcp_l0; \
        float o2 = O_ACC_INST.z * rcp_l1; \
        float o3 = O_ACC_INST.w * rcp_l1; \
        *reinterpret_cast<uint32_t*>(&smem_O[SWIZZLE_128(row_0, col_0)]) = pack_f32_to_h2(o0, o1); \
        *reinterpret_cast<uint32_t*>(&smem_O[SWIZZLE_128(row_1, col_0)]) = pack_f32_to_h2(o2, o3); \
    } while(0)

__global__ void __launch_bounds__(128, 2) prefill_attention_kernel(
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
    const int warp_m = warp_id;  

    extern __shared__ half smem_raw[];

    half* smem_Q = smem_raw; 
    const int q_offset = ((b_idx * seq_len + m_idx * TILE_M) * num_heads + h_idx) * HEAD_DIM;
    const bool is_valid_q = (m_idx * TILE_M + TILE_M <= seq_len);

    #pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        int idx = tid + i * 128;
        int row = idx / 16;
        int col = (idx % 16) * 8;
        if (is_valid_q || m_idx * TILE_M + row < seq_len) {
            cp_async_cg_16(&smem_Q[SWIZZLE_128(row, col)], &Q[q_offset + row * num_heads * HEAD_DIM + col]);
        } else {
            *reinterpret_cast<int4*>(&smem_Q[SWIZZLE_128(row, col)]) = make_int4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    uint4 q_frag_0, q_frag_1, q_frag_2, q_frag_3, q_frag_4, q_frag_5, q_frag_6, q_frag_7;
    int q_row = warp_m * 16 + (lane_id % 16);
    
    // Natively packing the scalar float multiplier to apply across half2 registers
    half2 scale_h2 = __floats2half2_rn(scale_log2, scale_log2);

    LDMATRIX_X4(q_frag_0, &smem_Q[SWIZZLE_128(q_row, 0 * 16 + (lane_id / 16) * 8)]); PRE_SCALE_Q_FRAG(q_frag_0);
    LDMATRIX_X4(q_frag_1, &smem_Q[SWIZZLE_128(q_row, 1 * 16 + (lane_id / 16) * 8)]); PRE_SCALE_Q_FRAG(q_frag_1);
    LDMATRIX_X4(q_frag_2, &smem_Q[SWIZZLE_128(q_row, 2 * 16 + (lane_id / 16) * 8)]); PRE_SCALE_Q_FRAG(q_frag_2);
    LDMATRIX_X4(q_frag_3, &smem_Q[SWIZZLE_128(q_row, 3 * 16 + (lane_id / 16) * 8)]); PRE_SCALE_Q_FRAG(q_frag_3);
    LDMATRIX_X4(q_frag_4, &smem_Q[SWIZZLE_128(q_row, 4 * 16 + (lane_id / 16) * 8)]); PRE_SCALE_Q_FRAG(q_frag_4);
    LDMATRIX_X4(q_frag_5, &smem_Q[SWIZZLE_128(q_row, 5 * 16 + (lane_id / 16) * 8)]); PRE_SCALE_Q_FRAG(q_frag_5);
    LDMATRIX_X4(q_frag_6, &smem_Q[SWIZZLE_128(q_row, 6 * 16 + (lane_id / 16) * 8)]); PRE_SCALE_Q_FRAG(q_frag_6);
    LDMATRIX_X4(q_frag_7, &smem_Q[SWIZZLE_128(q_row, 7 * 16 + (lane_id / 16) * 8)]); PRE_SCALE_Q_FRAG(q_frag_7);
    __syncthreads();

    half* smem_K_0 = smem_raw;  
    half* smem_K_1 = smem_raw + 4096;  
    half* smem_V_0 = smem_raw + 8192;  
    half* smem_V_1 = smem_raw + 12288;  
    half* smem_P   = smem_raw + 16384; 

    half* smem_K_load = smem_K_0;
    half* smem_V_load = smem_V_0;
    half* smem_K_compute = smem_K_0;
    half* smem_V_compute = smem_V_0;

    const int kv_base_offset = (b_idx * seq_len * num_kv_heads + kv_h_idx) * HEAD_DIM;
    const int kv_stride = num_kv_heads * HEAD_DIM;

    float m_i_0 = -1e30f, m_i_1 = -1e30f;
    float l_i_0 = 0.0f, l_i_1 = 0.0f;     
    
    float4 O_acc_0, O_acc_1, O_acc_2, O_acc_3, O_acc_4, O_acc_5, O_acc_6, O_acc_7;
    float4 O_acc_8, O_acc_9, O_acc_10, O_acc_11, O_acc_12, O_acc_13, O_acc_14, O_acc_15;
    ZERO_FLOAT4(O_acc_0); ZERO_FLOAT4(O_acc_1); ZERO_FLOAT4(O_acc_2); ZERO_FLOAT4(O_acc_3);
    ZERO_FLOAT4(O_acc_4); ZERO_FLOAT4(O_acc_5); ZERO_FLOAT4(O_acc_6); ZERO_FLOAT4(O_acc_7);
    ZERO_FLOAT4(O_acc_8); ZERO_FLOAT4(O_acc_9); ZERO_FLOAT4(O_acc_10); ZERO_FLOAT4(O_acc_11);
    ZERO_FLOAT4(O_acc_12); ZERO_FLOAT4(O_acc_13); ZERO_FLOAT4(O_acc_14); ZERO_FLOAT4(O_acc_15);

    const int max_kv_tile = min((m_idx + 1) * TILE_M, seq_len);
    const int num_kv_steps = (max_kv_tile + TILE_N - 1) / TILE_N;
    const int num_full_steps = (m_idx * TILE_M) / TILE_N;

    int stage = 0;

    if (0 < num_kv_steps) {
        LOAD_KV_TILE(0, smem_K_load, smem_V_load);
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
        smem_K_load = smem_K_1;
        smem_V_load = smem_V_1;
    }

    for (int k_tile = 0; k_tile < num_kv_steps; ++k_tile) {
        int next_tile = k_tile + 1;
        bool has_next = next_tile < num_kv_steps;
        
        if (has_next) {
            if (stage == 0) {
                LOAD_KV_TILE(next_tile, smem_K_1, smem_V_1);
            } else {
                LOAD_KV_TILE(next_tile, smem_K_0, smem_V_0);
            }
            cp_async_commit();
        }

        half* compute_K = (stage == 0) ? smem_K_0 : smem_K_1;
        half* compute_V = (stage == 0) ? smem_V_0 : smem_V_1;

        float4 S_acc_0, S_acc_1, S_acc_2, S_acc_3;
        ZERO_FLOAT4(S_acc_0); ZERO_FLOAT4(S_acc_1); ZERO_FLOAT4(S_acc_2); ZERO_FLOAT4(S_acc_3);

        COMPUTE_S_ACC_STEP(0, q_frag_0);
        COMPUTE_S_ACC_STEP(1, q_frag_1);
        COMPUTE_S_ACC_STEP(2, q_frag_2);
        COMPUTE_S_ACC_STEP(3, q_frag_3);
        COMPUTE_S_ACC_STEP(4, q_frag_4);
        COMPUTE_S_ACC_STEP(5, q_frag_5);
        COMPUTE_S_ACC_STEP(6, q_frag_6);
        COMPUTE_S_ACC_STEP(7, q_frag_7);

        float local_max_0 = -1e30f, local_max_1 = -1e30f;
        const bool is_causal_tile = (k_tile >= num_full_steps);

        int r0 = m_idx * TILE_M + warp_m * 16 + (lane_id / 4);
        int r1 = r0 + 8;

        if (is_causal_tile) {
            MASK_S_ACC(0, S_acc_0); MASK_S_ACC(1, S_acc_1);
            MASK_S_ACC(2, S_acc_2); MASK_S_ACC(3, S_acc_3);
        }
        
        // Because Q was pre-scaled during load, we completely skip FP32 mults here
        UPDATE_MAX(S_acc_0); UPDATE_MAX(S_acc_1);
        UPDATE_MAX(S_acc_2); UPDATE_MAX(S_acc_3);

        #pragma unroll 2
        for (int mask = 1; mask <= 2; mask *= 2) {
            local_max_0 = fmaxf(local_max_0, __shfl_xor_sync(0xffffffff, local_max_0, mask));
            local_max_1 = fmaxf(local_max_1, __shfl_xor_sync(0xffffffff, local_max_1, mask));
        }

        float m_new_0 = fmaxf(m_i_0, local_max_0);
        float m_new_1 = fmaxf(m_i_1, local_max_1);
        float alpha_0 = fast_exp2(m_i_0 - m_new_0);
        float alpha_1 = fast_exp2(m_i_1 - m_new_1);

        // Uses a warp-wide sync to totally skip 64 FP32 multiplies if the running max has stabilized
        bool need_scale_0 = __any_sync(0xffffffff, alpha_0 != 1.0f);
        bool need_scale_1 = __any_sync(0xffffffff, alpha_1 != 1.0f);

        SCALE_O_ACC(O_acc_0); SCALE_O_ACC(O_acc_1); SCALE_O_ACC(O_acc_2); SCALE_O_ACC(O_acc_3);
        SCALE_O_ACC(O_acc_4); SCALE_O_ACC(O_acc_5); SCALE_O_ACC(O_acc_6); SCALE_O_ACC(O_acc_7);
        SCALE_O_ACC(O_acc_8); SCALE_O_ACC(O_acc_9); SCALE_O_ACC(O_acc_10); SCALE_O_ACC(O_acc_11);
        SCALE_O_ACC(O_acc_12); SCALE_O_ACC(O_acc_13); SCALE_O_ACC(O_acc_14); SCALE_O_ACC(O_acc_15);

        float P_sum_0 = 0.0f, P_sum_1 = 0.0f;
        int p_row0 = warp_m * 16 + (lane_id / 4);
        int p_row1 = p_row0 + 8;
        
        COMPUTE_P_AND_STORE(0, S_acc_0);
        COMPUTE_P_AND_STORE(1, S_acc_1);
        COMPUTE_P_AND_STORE(2, S_acc_2);
        COMPUTE_P_AND_STORE(3, S_acc_3);

        #pragma unroll 2
        for (int mask = 1; mask <= 2; mask *= 2) {
            P_sum_0 += __shfl_xor_sync(0xffffffff, P_sum_0, mask);
            P_sum_1 += __shfl_xor_sync(0xffffffff, P_sum_1, mask);
        }

        l_i_0 = fmaf(l_i_0, alpha_0, P_sum_0);
        l_i_1 = fmaf(l_i_1, alpha_1, P_sum_1);
        m_i_0 = m_new_0;
        m_i_1 = m_new_1;

        __syncwarp(); 

        #pragma unroll 2
        for (int k_step = 0; k_step < 2; ++k_step) { 
            uint4 p_frag;
            int p_row = warp_m * 16 + (lane_id % 16);
            int p_col = k_step * 16 + (lane_id / 16) * 8;
            LDMATRIX_X4(p_frag, &smem_P[SWIZZLE_64(p_row, p_col)]);
            
            int v_row = k_step * 16 + (lane_id % 16);
            
            COMPUTE_O_ACC_GROUP(0, O_acc_0, O_acc_1, O_acc_2, O_acc_3);
            COMPUTE_O_ACC_GROUP(4, O_acc_4, O_acc_5, O_acc_6, O_acc_7);
            COMPUTE_O_ACC_GROUP(8, O_acc_8, O_acc_9, O_acc_10, O_acc_11);
            COMPUTE_O_ACC_GROUP(12, O_acc_12, O_acc_13, O_acc_14, O_acc_15);
        }

        if (has_next) {
            cp_async_wait<0>();  
            __syncthreads();    
            stage ^= 1; 
        }
    }

    __syncthreads();  

    half* smem_O = smem_raw;
    float rcp_l0 = (l_i_0 > 0.0f) ? fast_rcp(l_i_0) : 0.0f;
    float rcp_l1 = (l_i_1 > 0.0f) ? fast_rcp(l_i_1) : 0.0f;

    int row_0 = warp_m * 16 + (lane_id / 4);
    int row_1 = row_0 + 8;
        
    STORE_O_ACC(0, O_acc_0); STORE_O_ACC(1, O_acc_1);
    STORE_O_ACC(2, O_acc_2); STORE_O_ACC(3, O_acc_3);
    STORE_O_ACC(4, O_acc_4); STORE_O_ACC(5, O_acc_5);
    STORE_O_ACC(6, O_acc_6); STORE_O_ACC(7, O_acc_7);
    STORE_O_ACC(8, O_acc_8); STORE_O_ACC(9, O_acc_9);
    STORE_O_ACC(10, O_acc_10); STORE_O_ACC(11, O_acc_11);
    STORE_O_ACC(12, O_acc_12); STORE_O_ACC(13, O_acc_13);
    STORE_O_ACC(14, O_acc_14); STORE_O_ACC(15, O_acc_15);
    
    __syncthreads();  
    
    #pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        int idx = tid + i * 128;   
        int row = idx / 16;
        int col = (idx % 16) * 8;
        if (m_idx * TILE_M + row < seq_len) {
            int out_off = ((b_idx * seq_len + m_idx * TILE_M + row) * num_heads + h_idx) * HEAD_DIM + col;
            *reinterpret_cast<int4*>(&O[out_off]) = *reinterpret_cast<int4*>(&smem_O[SWIZZLE_128(row, col)]);
        }
    }
}

torch::Tensor prefill_attention_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, float scale,
    c10::optional<torch::Tensor> out_opt = c10::nullopt
) {
    const int batch_size = q.size(0);
    const int seq_len = q.size(1);
    const int num_heads = q.size(2);
    const int num_kv_heads = k.size(2);
    const int head_dim = q.size(3);

    torch::Tensor out;
    if (out_opt.has_value()) {
        out = out_opt.value();
    } else {
        auto options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
        out = torch::empty({batch_size, seq_len, num_heads, head_dim}, options);
    }

    dim3 grid((seq_len + TILE_M - 1) / TILE_M, num_heads, batch_size);
    dim3 block(128); 

    size_t smem_bytes = 42 * 1024; 

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
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
