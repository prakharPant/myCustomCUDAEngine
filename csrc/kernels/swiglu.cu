#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

__global__ void swiglu_kernel_fp16(
    const __half
        *__restrict__ gate_up, // Contiguous memory containing [gate, up]
    __half *__restrict__ output, int intermediate_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= intermediate_size)
    return;

  int token_idx = idx / (intermediate_size / 2);
  int col_idx = idx % (intermediate_size / 2);

  int gate_offset = token_idx * intermediate_size + col_idx;
  int up_offset = gate_offset + (intermediate_size / 2);

  float g = __half2float(gate_up[gate_offset]);
  float u = __half2float(gate_up[up_offset]);

  float res = silu(g) * u;
  output[token_idx * (intermediate_size / 2) + col_idx] = __float2half(res);
}

torch::Tensor swiglu_cuda(torch::Tensor gate_up) {
  TORCH_CHECK(gate_up.is_cuda(), "Input must be CUDA");
  TORCH_CHECK(gate_up.scalar_type() == torch::kHalf, "Input must be FP16");

  int total_elements = gate_up.numel();
  int intermediate_size = gate_up.size(-1);
  int half_size = intermediate_size / 2;
  int num_tokens = total_elements / intermediate_size;

  auto output_shape = gate_up.sizes().vec();
  output_shape.back() = half_size;
  auto output = torch::empty(output_shape, gate_up.options());

  int num_threads = 256;
  int num_blocks = (num_tokens * half_size + num_threads - 1) / num_threads;

  swiglu_kernel_fp16<<<num_blocks, num_threads>>>(
      reinterpret_cast<const __half *>(gate_up.data_ptr<at::Half>()),
      reinterpret_cast<__half *>(output.data_ptr<at::Half>()),
      intermediate_size);

  return output;
}
