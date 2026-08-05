#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Helper structure for 128-bit memory alignment (8 x FP16 = 16 bytes)
struct alignas(16) Float4Packed {
  uint4 raw;
};

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceSum(val);
  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
  if (wid == 0)
    val = warpReduceSum(val);
  return val;
}

__global__ void rmsnorm_kernel_vectorized_128bit(
    const __half *__restrict__ input, const __half *__restrict__ weight,
    __half *__restrict__ output, int hidden_size, float eps) {
  int row_idx = blockIdx.x;

  // 128-bit pointer casting (8 x FP16 elements per uint4)
  const uint4 *row_in128 =
      reinterpret_cast<const uint4 *>(input + row_idx * hidden_size);
  const uint4 *weight128 = reinterpret_cast<const uint4 *>(weight);
  uint4 *row_out128 = reinterpret_cast<uint4 *>(output + row_idx * hidden_size);

  int vec_size = hidden_size /
                 8; // Assuming hidden_size is divisible by 8 (4096 / 8 = 512)

    // Register storage to avoid SECOND global memory pass
    // With blockDim.x = 256, each thread handles 512 / 256 = 2 vectors (16 FP16 elements)
    constexpr int MAX_VECS_PER_THREAD = 4;
    uint4 reg_in[MAX_VECS_PER_THREAD];
    uint4 reg_w[MAX_VECS_PER_THREAD];

  float variance = 0.0f;
  int vec_idx = threadIdx.x;
  int reg_i = 0;

  // --- PASS 1: Single Global Memory Load + Local Register Cache + Accumulate
  // Variance ---
  for (; vec_idx < vec_size && reg_i < MAX_VECS_PER_THREAD;
       vec_idx += blockDim.x, reg_i++) {
    reg_in[reg_i] = row_in128[vec_idx]; // 128-bit LDG.E.128 instruction
    reg_w[reg_i] = weight128[vec_idx];  // 128-bit LDG.E.128 instruction

    const __half2 *h2_in = reinterpret_cast<const __half2 *>(&reg_in[reg_i]);

#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 f2 = __half22float2(h2_in[k]);
      variance += f2.x * f2.x + f2.y * f2.y;
    }
  }

  // Inter-thread Block Reduction
  variance = blockReduceSum(variance);

  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf((variance / hidden_size) + eps);
  }
  __syncthreads();

  float inv_rms = s_inv_rms;

  // --- PASS 2: Compute Normalization Directly From Registers & 128-bit Global
  // Store ---
  vec_idx = threadIdx.x;
  for (int i = 0; i < reg_i; i++, vec_idx += blockDim.x) {
    uint4 out_u4;
    const __half2 *h2_in = reinterpret_cast<const __half2 *>(&reg_in[i]);
    const __half2 *h2_w = reinterpret_cast<const __half2 *>(&reg_w[i]);
    __half2 *h2_out = reinterpret_cast<__half2 *>(&out_u4);

#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 f_in = __half22float2(h2_in[k]);
      float2 f_w = __half22float2(h2_w[k]);

      f_in.x *= inv_rms * f_w.x;
      f_in.y *= inv_rms * f_w.y;

      h2_out[k] = __floats2half2_rn(f_in.x, f_in.y);
    }

    row_out128[vec_idx] = out_u4; // 128-bit STG.E.128 instruction
  }
}

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight,
                           float eps) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda());
  TORCH_CHECK(input.scalar_type() == torch::kHalf);

  auto hidden_size = input.size(-1);
  auto num_tokens = input.numel() / hidden_size;
  auto output = torch::empty_like(input);

  dim3 grid(num_tokens);
  dim3 block(256); // 256 threads * 2 iterations * 8 elements = 4096 hidden size
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rmsnorm_kernel_vectorized_128bit<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half *>(weight.data_ptr<at::Half>()),
      reinterpret_cast<__half *>(output.data_ptr<at::Half>()), hidden_size,
      eps);

  return output;
}
