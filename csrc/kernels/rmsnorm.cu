#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Warp reduction helper
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction helper
__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Shared memory for up to 1024 threads (32 warps)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void rmsnorm_kernel_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int hidden_size,
    float eps
) {
    int row_idx = blockIdx.x;

    const __half* row_in = input + row_idx * hidden_size;
    __half* row_out = output + row_idx * hidden_size;

    // Treat every two half values as one half2
    const half2* row_in2 = reinterpret_cast<const half2*>(row_in);
    const half2* weight2 = reinterpret_cast<const half2*>(weight);
    half2* row_out2 = reinterpret_cast<half2*>(row_out);

    int hidden_size2 = hidden_size / 2;

    //---------------------------------------------------------
    // Pass 1 : Compute variance
    //---------------------------------------------------------

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const int stride = blockDim.x;
    const int unroll = 4;

    int i = threadIdx.x;

    // Main unrolled loop
    for (; i + (unroll - 1) * stride < hidden_size2;
           i += stride * unroll)
    {
        half2 h0 = row_in2[i];
        half2 h1 = row_in2[i + stride];
        half2 h2 = row_in2[i + 2 * stride];
        half2 h3 = row_in2[i + 3 * stride];

	float2 f0 = __half22float2(h0);
	float2 f1 = __half22float2(h1);
        float2 f2 = __half22float2(h2);
        float2 f3 = __half22float2(h3);

        sum0 += f0.x * f0.x + f0.y * f0.y;
        sum1 += f1.x * f1.x + f1.y * f1.y;
        sum2 += f2.x * f2.x + f2.y * f2.y;
        sum3 += f3.x * f3.x + f3.y * f3.y;
    }

    // Tail loop
    for (; i < hidden_size2; i += stride)
    {
        half2 h = row_in2[i];
        float2 f = __half22float2(h);

        sum0 += f.x * f.x + f.y * f.y;
    }

    float variance = sum0 + sum1 + sum2 + sum3;

    // Handle odd hidden size
    if ((hidden_size & 1) &&
        threadIdx.x == 0)
    {
        float v = __half2float(row_in[hidden_size - 1]);
        variance += v * v;
    }

    variance = blockReduceSum(variance);

    //---------------------------------------------------------
    // Compute inverse RMS
    //---------------------------------------------------------

    __shared__ float s_inv_rms;

    if (threadIdx.x == 0)
    {
        s_inv_rms = rsqrtf((variance / hidden_size) + eps);
    }

    __syncthreads();

    float inv_rms = s_inv_rms;

    //---------------------------------------------------------
    // Pass 2 : Normalize
    //---------------------------------------------------------

    int j = threadIdx.x;

    for (; j + 3 * stride < hidden_size2;
           j += stride * 4)
    {
        half2 in0 = row_in2[j];
        half2 in1 = row_in2[j + stride];
        half2 in2 = row_in2[j + 2 * stride];
        half2 in3 = row_in2[j + 3 * stride];

        half2 wt0 = weight2[j];
        half2 wt1 = weight2[j + stride];
        half2 wt2 = weight2[j + 2 * stride];
        half2 wt3 = weight2[j + 3 * stride];

        float2 v0 = __half22float2(in0);
        float2 v1 = __half22float2(in1);
        float2 v2 = __half22float2(in2);
        float2 v3 = __half22float2(in3);

        float2 w0 = __half22float2(wt0);
        float2 w1 = __half22float2(wt1);
        float2 w2 = __half22float2(wt2);
        float2 w3 = __half22float2(wt3);

        v0.x *= inv_rms * w0.x;
        v0.y *= inv_rms * w0.y;

        v1.x *= inv_rms * w1.x;
        v1.y *= inv_rms * w1.y;

        v2.x *= inv_rms * w2.x;
        v2.y *= inv_rms * w2.y;

        v3.x *= inv_rms * w3.x;
        v3.y *= inv_rms * w3.y;

        row_out2[j] = __floats2half2_rn(v0.x, v0.y);
        row_out2[j + stride] = __floats2half2_rn(v1.x, v1.y);
        row_out2[j + 2 * stride] = __floats2half2_rn(v2.x, v2.y);
        row_out2[j + 3 * stride] = __floats2half2_rn(v3.x, v3.y);
    }

    for (; j < hidden_size2; j += stride)
    {
        half2 in = row_in2[j];
        half2 wt = weight2[j];

        float2 vin = __half22float2(in);
        float2 vw  = __half22float2(wt);

        vin.x *= inv_rms * vw.x;
        vin.y *= inv_rms * vw.y;

        row_out2[j] = __floats2half2_rn(vin.x, vin.y);
    }


    // Handle odd element
    if ((hidden_size & 1) &&
        threadIdx.x == 0)
    {
        int i = hidden_size - 1;

        float val = __half2float(row_in[i]);
        float w   = __half2float(weight[i]);

        row_out[i] = __float2half(val * inv_rms * w);
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, float eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "Input must be FP16");

    auto hidden_size = input.size(-1);
    auto num_tokens = input.numel() / hidden_size;

    auto output = torch::empty_like(input);

    dim3 grid(num_tokens);
    dim3 block(std::min((int)hidden_size, 256));

    rmsnorm_kernel_fp16<<<grid, block>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        hidden_size,
        eps
    );

    return output;
}
