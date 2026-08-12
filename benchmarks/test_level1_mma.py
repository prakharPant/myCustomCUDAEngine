import torch
import my_cuda_engine_cpp

def test_level1_mma():
    device = "cuda"
    torch.manual_seed(42)

    A = torch.randn(16, 16, dtype=torch.float16, device=device)
    B = torch.randn(16, 8, dtype=torch.float16, device=device)

    # Custom CUDA Execution
    C_custom = my_cuda_engine_cpp.run_level1_mma_microtest(A, B)

    # PyTorch Reference Execution
    C_pt = torch.matmul(A.float(), B.float())

    max_diff = (C_custom - C_pt).abs().max().item()
    mean_diff = (C_custom - C_pt).abs().mean().item()
    is_correct = torch.allclose(C_custom, C_pt, atol=1e-3, rtol=1e-3)

    print("=========================================================")
    print(" 🧪 LEVEL 1: PTX ldmatrix + mma.sync Microtest Verification")
    print("=========================================================")
    print(f"Status   : {'PASSED' if is_correct else 'FAILED'}")
    print(f"Max Diff : {max_diff:.6e}")
    print(f"Mean Diff: {mean_diff:.6e}")
    print("=========================================================")

    if not is_correct:
        print("\nC_custom:\n", C_custom)
        print("\nC_pt:\n", C_pt)

if __name__ == "__main__":
    test_level1_mma()
