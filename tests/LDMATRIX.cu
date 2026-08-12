/*
The following code correctly loads the Q and K matrices to registers and preforms a matmul operation Q@K^T then correctly reads the output and compares with the naive method for numerical accuracy.
*/
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdint.h>

// --- MMA helpers ---
__device__ __forceinline__ void ldmatrix_x4(uint32_t* R, const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3])
        : "r"(smem_addr)
    );
}

// FIX: Dropped the .trans modifier here
__device__ __forceinline__ void ldmatrix_x2(uint32_t* R, const void* smem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
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

// --- Kernel: compute S = Q @ K^T ---
__global__ void mma_test_kernel(const half* __restrict__ Q,
                                const half* __restrict__ K,
                                float* __restrict__ S) {
    const int tid     = threadIdx.x;
    const int lane_id = tid % 32;

    __shared__ half smem_Q[16*16];
    __shared__ half smem_K[16*16];

    // cooperative load
    for (int i = tid; i < 16*16; i += 32) {
        smem_Q[i] = Q[i];
        smem_K[i] = K[i];
    }
    __syncthreads();

    // load Q fragment (Your original, correct math)
    uint32_t Q_frag[4];
    {
        int q_row = lane_id % 16;
        int q_col = (lane_id / 16) * 8;
        ldmatrix_x4(Q_frag, &smem_Q[q_row * 16 + q_col]);
    }

    // load K fragments (Your original, correct math, using standard ldmatrix_x2)
    uint32_t K0_frag[2], K1_frag[2];
    {
        int k0_row = lane_id % 8;
        int k0_col = (lane_id / 8) * 8;
        ldmatrix_x2(K0_frag, &smem_K[k0_row * 16 + k0_col]);

        int k1_row = 8 + (lane_id % 8);
        int k1_col = (lane_id / 8) * 8;
        ldmatrix_x2(K1_frag, &smem_K[k1_row * 16 + k1_col]);
    }

    float S0[4] = {0.f,0.f,0.f,0.f};
    float S1[4] = {0.f,0.f,0.f,0.f};
    mma_m16n8k16_f32(S0, Q_frag, K0_frag);
    mma_m16n8k16_f32(S1, Q_frag, K1_frag);

    // Map warp lane to row and column offsets
    int row0 = lane_id / 4;        // Rows 0..7
    int row1 = row0 + 8;           // Rows 8..15
    int col0 = (lane_id % 4) * 2;  // Cols 0..7  (for S0 block)
    int col1 = col0 + 8;           // Cols 8..15 (for S1 block)

    // Write S0 (Left 16x8 half of matrix S)
    S[row0 * 16 + col0]     = S0[0];
    S[row0 * 16 + col0 + 1] = S0[1];
    S[row1 * 16 + col0]     = S0[2];
    S[row1 * 16 + col0 + 1] = S0[3];

    // Write S1 (Right 16x8 half of matrix S)
    S[row0 * 16 + col1]     = S1[0];
    S[row0 * 16 + col1 + 1] = S1[1];
    S[row1 * 16 + col1]     = S1[2];
    S[row1 * 16 + col1 + 1] = S1[3];
}

// --- CPU reference ---
void cpu_matmul(const half* Q, const half* K, float* S) {
    for (int i=0;i<16;i++) {
        for (int j=0;j<16;j++) {
            float acc = 0.f;
            for (int k=0;k<16;k++) {
                acc += __half2float(Q[i*16+k]) * __half2float(K[j*16+k]);
            }
            S[i*16+j] = acc;
        }
    }
}

int main() {
    half hQ[16*16], hK[16*16];
    for (int i=0;i<16*16;i++) {
        hQ[i] = __float2half((float)(rand()%5));
        hK[i] = __float2half((float)(rand()%5));
    }

    half *dQ,*dK;
    float *dS,*hS_mma,*hS_ref;
    hS_mma = (float*)malloc(16*16*sizeof(float));
    hS_ref = (float*)malloc(16*16*sizeof(float));

    cudaMalloc(&dQ,16*16*sizeof(half));
    cudaMalloc(&dK,16*16*sizeof(half));
    cudaMalloc(&dS,16*16*sizeof(float));
    cudaMemcpy(dQ,hQ,16*16*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(dK,hK,16*16*sizeof(half),cudaMemcpyHostToDevice);

    mma_test_kernel<<<1,32>>>(dQ,dK,dS);
    cudaMemcpy(hS_mma,dS,16*16*sizeof(float),cudaMemcpyDeviceToHost);

    cpu_matmul(hQ,hK,hS_ref);

    // Print CPU reference result
    printf("\nReference S = Q @ K^T (CPU, float):\n");
    for (int i=0;i<16;i++) {
        for (int j=0;j<16;j++) {
            printf("%8.2f ", hS_ref[i*16+j]);
        }
        printf("\n");
    }

    // Print MMA result
    printf("\nMMA result S (GPU, float):\n");
    for (int i=0;i<16;i++) {
        for (int j=0;j<16;j++) {
            printf("%8.2f ", hS_mma[i*16+j]);
        }
        printf("\n");
    }    

    // Compare
    double max_err=0;
    for (int i=0;i<16*16;i++) {
        double err = fabs(hS_mma[i]-hS_ref[i]);
        if (err>max_err) max_err=err;
    }
    printf("Max abs error = %f\n",max_err);

    cudaFree(dQ); cudaFree(dK); cudaFree(dS);
    free(hS_mma); free(hS_ref);
    return 0;
}
