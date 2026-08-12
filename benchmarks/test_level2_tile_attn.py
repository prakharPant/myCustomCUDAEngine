import math
import torch
import my_cuda_engine_cpp

def test_level2_tile_attn():
    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(42)

    Q = torch.randn(16, 16, dtype=dtype, device=device)
    K = torch.randn(16, 16, dtype=dtype, device=device)
    V = torch.randn(16, 16, dtype=dtype, device=device)
    scale = 1.0 / math.sqrt(16)

    # Custom CUDA Tile Attention Kernel
    O_custom = my_cuda_engine_cpp.run_level2_tile_attn(Q, K, V, scale)

    # PyTorch Reference Execution
    scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * scale
    mask = torch.triu(torch.full((16, 16), float('-inf'), device=device), diagonal=1)
    scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    O_pt = torch.matmul(probs, V.float()).to(dtype)

    max_diff = (O_custom - O_pt).abs().max().item()
    mean_diff = (O_custom - O_pt).abs().mean().item()
    is_correct = torch.allclose(O_custom, O_pt, atol=1e-2, rtol=1e-2)

    print("=========================================================")
    print(" 🧪 LEVEL 2: Single-Warp Fused Tile Attention Verification")
    print("=========================================================")
    print(f"Status   : {'PASSED' if is_correct else 'FAILED'}")
    print(f"Max Diff : {max_diff:.6e}")
    print(f"Mean Diff: {mean_diff:.6e}")
    print("=========================================================")

    if not is_correct:
        print("\nO_custom:\n", O_custom)
        print("\nO_pt:\n", O_pt)

if __name__ == "__main__":
    test_level2_tile_attn()
