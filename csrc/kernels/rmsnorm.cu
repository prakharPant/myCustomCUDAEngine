#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Warp reduction helper
// Memory ops are expensive; register ops are cheap.
// This method minimizes memory traffic and synchronization.
// It scales naturally: 32 threads reduced in 5 steps instead of 32 steps.
//
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// Block reduction helper
__device__ __forceinline__ float blockReduceSum(float val) {
  static __shared__ float
      shared[32]; // Shared memory for up to 1024 threads (32 warps)
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

__global__ void rmsnorm_kernel_fp16(const __half *__restrict__ input,
                                    const __half *__restrict__ weight,
                                    __half *__restrict__ output,
                                    int hidden_size, float eps) {
  int row_idx = blockIdx.x;
  const __half *row_in = input + row_idx * hidden_size;
  __half *row_out = output + row_idx * hidden_size;

  float variance = 0.0f;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = __half2float(row_in[i]);
    variance += val * val;
  }

  variance = blockReduceSum(variance);

  __shared__ float s_inv_rms;
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf((variance / hidden_size) + eps);
  }
  __syncthreads();

  float inv_rms = s_inv_rms;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float val = __half2float(row_in[i]);
    float w = __half2float(weight[i]);
    row_out[i] = __float2half(val * inv_rms * w);
  }
}

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight,
                           float eps) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kHalf, "Input must be FP16");

  auto hidden_size = input.size(-1);
  auto num_tokens = input.numel() / hidden_size;

  auto output = torch::empty_like(input);

  dim3 grid(num_tokens);
  dim3 block(std::min((int)hidden_size, 1024));

  rmsnorm_kernel_fp16<<<grid, block>>>(
      reinterpret_cast<const __half *>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half *>(weight.data_ptr<at::Half>()),
      reinterpret_cast<__half *>(output.data_ptr<at::Half>()), hidden_size,
      eps);

  return output;
}
