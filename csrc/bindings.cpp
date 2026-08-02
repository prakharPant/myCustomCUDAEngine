#include <torch/extension.h>

// Forward declarations
torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight,
                           float eps);
torch::Tensor swiglu_cuda(torch::Tensor gate_up);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm_cuda, "Custom Fused RMSNorm kernel (CUDA)");
  m.def("swiglu", &swiglu_cuda, "Custom Fused SwiGLU kernel (CUDA)");
}
