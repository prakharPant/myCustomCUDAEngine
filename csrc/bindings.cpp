#include <torch/extension.h>

// Forward declarations
torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight,
                           float eps);
torch::Tensor swiglu_cuda(torch::Tensor gate_up);
void paged_kv_store_cuda(torch::Tensor key, torch::Tensor value, torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor slot_mapping);
torch::Tensor paged_attention_decode_cuda(torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor block_tables, torch::Tensor context_lens, int max_blocks_per_seq, int block_size, float scale);
void rope_inplace_cuda(torch::Tensor query,
                       torch::Tensor key,
                       torch::Tensor cos_table,
                       torch::Tensor sin_table,
                       torch::Tensor positions);
torch::Tensor prefill_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    float scale,
    c10::optional<torch::Tensor> out_opt = c10::nullopt
);
torch::Tensor run_level1_mma_microtest(torch::Tensor A, torch::Tensor B);
torch::Tensor run_level2_tile_attn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm_cuda, "Custom Fused RMSNorm kernel (CUDA)");
  m.def("swiglu", &swiglu_cuda, "Custom Fused SwiGLU kernel (CUDA)");
  m.def("paged_kv_store", &paged_kv_store_cuda, "Scatter K/V vectors into physical block pages (CUDA)");
  m.def("paged_attention_decode", &paged_attention_decode_cuda, "Paged FlashAttention Decode kernel (CUDA)");
  m.def("rope_inplace", &rope_inplace_cuda, "Fused In-place RoPE kernel");
  m.def("prefill_attention", &prefill_attention_cuda, "Custom Causal Prefill Attention (CUDA)", py::arg("q"),py::arg("k"),py::arg("v"),py::arg("scale"),py::arg("out") = nullptr);
  m.def("run_level1_mma_microtest", &run_level1_mma_microtest, "Level 1 MMA Microtest (CUDA)");
  m.def("run_level2_tile_attn", &run_level2_tile_attn, "Level 2 Tile Attention (CUDA)");
}
