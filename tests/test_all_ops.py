import torch
import my_cuda_engine_cpp

def test_op(name, fn_captured_and_eager):
    s = torch.cuda.Stream()
    
    # 1. Capture stream-synchronized graph
    with torch.cuda.stream(s):
        for _ in range(3): fn_captured_and_eager(is_replay=False) # Warmup
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out_graph = fn_captured_and_eager(is_replay=False)
    s.synchronize()

    # 2. Replay Graph
    g.replay()
    torch.cuda.synchronize()

    # 3. Run Eager Mode on fresh tensors
    out_eager = fn_captured_and_eager(is_replay=True)
    torch.cuda.synchronize()

    diff = (out_eager - out_graph).abs().max().item()
    print(f"[{'PASS' if diff < 1e-3 else 'FAIL'}] {name:<22} Max Diff: {diff:.6f}")

# Setup Inputs
d, device = torch.float16, "cuda"
x = torch.randn(1, 3072, dtype=d, device=device)
w = torch.ones(3072, dtype=d, device=device)
q = torch.randn(1, 1, 24, 128, dtype=d, device=device)
k = torch.randn(1, 1, 8, 128, dtype=d, device=device)
cos = torch.randn(4096, 64, dtype=torch.float32, device=device)
sin = torch.randn(4096, 64, dtype=torch.float32, device=device)
pos = torch.zeros(1, 1, dtype=torch.int32, device=device)
kc = torch.zeros(10, 16, 8, 128, dtype=d, device=device)
vc = torch.zeros(10, 16, 8, 128, dtype=d, device=device)
slot = torch.tensor([0], dtype=torch.int32, device=device)
bt = torch.tensor([[0]], dtype=torch.int32, device=device)
cl = torch.tensor([1], dtype=torch.int32, device=device)
g_up = torch.randn(1, 16384, dtype=d, device=device)

# --- 1. Test RMSNorm ---
static_out_rms = torch.empty_like(x)
test_op("rmsnorm", lambda is_replay: my_cuda_engine_cpp.rmsnorm(x, w, 1e-5))

# --- 2. Test RoPE Inplace ---
q_rope, k_rope = q.clone(), k.clone()
def run_rope(is_replay):
    if is_replay: q_rope.copy_(q); k_rope.copy_(k)
    my_cuda_engine_cpp.rope_inplace(q_rope, k_rope, cos, sin, pos)
    return q_rope
test_op("rope_inplace", run_rope)

# --- 3. Test Paged KV Store ---
def run_kv(is_replay):
    if is_replay: kc.zero_(); vc.zero_()
    my_cuda_engine_cpp.paged_kv_store(k.squeeze(1), k.squeeze(1), kc, vc, slot)
    return kc
test_op("paged_kv_store", run_kv)

# --- 4. Test Paged Attention Decode ---
test_op("paged_attn_decode", lambda is_replay: my_cuda_engine_cpp.paged_attention_decode(q.squeeze(1), kc, vc, bt, cl, 1, 16, 0.1))

# --- 5. Test SwiGLU ---
test_op("swiglu", lambda is_replay: my_cuda_engine_cpp.swiglu(g_up))
