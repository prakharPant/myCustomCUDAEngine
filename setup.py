import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Optimization flags for modern NVCC compilers
extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-gencode=arch=compute_80,code=sm_80",  # Ampere (A100)
        "-gencode=arch=compute_89,code=sm_89",  # Ada (RTX 4090)
        "-gencode=arch=compute_90,code=sm_90",  # Hopper (H100)
    ],
}

setup(
    name="my_cuda_engine_cpp",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="my_cuda_engine_cpp",
            sources=[
                "csrc/bindings.cpp",
                "csrc/kernels/rmsnorm.cu",
                "csrc/kernels/swiglu.cu",
            ],
            include_dirs=[os.path.abspath("csrc/includes")],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
