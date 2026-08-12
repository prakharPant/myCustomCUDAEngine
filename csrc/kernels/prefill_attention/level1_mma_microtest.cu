#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Matrix A: Loads 16x16 FP16 from SMEM into 4x uint32_t registers
__device__ __forceinline__ void ldmatrix_x4(uint32_t* R, const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
        : "r"(smem_addr)
    );
}

// Matrix B: Loads 16x8 FP16 from SMEM and transposes into 2x uint32_t registers
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* R, const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(R[0]), "=r"(R[1])
        : "r"(smem_addr)
    );
}

// Tensor Core GEMM: A[16x16] * B[16x8] = C[16x8]
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

__global__ void level1_mma_microtest_kernel(
    const half* __restrict__ A,  // [16, 16]
    const half* __restrict__ B,  // [16, 8]
    float* __restrict__ C        // [16, 8]
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;

    __shared__ half smem_A[16 * 16];
    __shared__ half smem_B[16 * 8];

    // 1. Cooperative load A and B into SMEM
    for (int i = tid; i < 16 * 16; i += 32) {
        smem_A[i] = A[i];
    }
    for (int i = tid; i < 16 * 8; i += 32) {
        smem_B[i] = B[i];
    }
    __syncthreads();

    // 2. Load A [16x16] via ldmatrix.x4
    uint32_t A_frag[4];
    int a_row = lane_id % 16;
    int a_col = (lane_id / 16) * 8;
    ldmatrix_x4(A_frag, &smem_A[a_row * 16 + a_col]);

    // 3. Load B [16x8] via ldmatrix.x2.trans
    uint32_t B_frag[2];
    int b_row = lane_id % 16;
    int b_col = 0;
    ldmatrix_x2_trans(B_frag, &smem_B[b_row * 8 + b_col]);

    // 4. Compute Tensor Core GEMM
    float D_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_m16n8k16_f32(D_acc, A_frag, B_frag);

    // 5. Unpack D registers to matrix C [16x8] according to PTX ISA spec
    int group = lane_id / 4;
    int tid_in_group = lane_id % 4;

    int row0 = group;
    int row1 = group + 8;

    int col0 = tid_in_group * 2;
    int col1 = col0 + 1;

    C[row0 * 8 + col0] = D_acc[0];
    C[row0 * 8 + col1] = D_acc[1];
    C[row1 * 8 + col0] = D_acc[2];
    C[row1 * 8 + col1] = D_acc[3];
}

torch::Tensor run_level1_mma_microtest(torch::Tensor A, torch::Tensor B) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor C = torch::zeros({16, 8}, options);

    level1_mma_microtest_kernel<<<1, 32>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>()
    );

    return C;
}
