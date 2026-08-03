import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-lineinfo",          # <-- 1. keep source->SASS mapping
        "-Xptxas=-v",         # <-- 2. print regs/smem per kernel
        "--use_fast_math",
        # For development, ONLY compile for YOUR GPU. 10x faster build.
        "-gencode=arch=compute_86,code=sm_86",
        # For release build, add others back:
        # "-gencode=arch=compute_80,code=sm_80",
        # "-gencode=arch=compute_89,code=sm_89",
        # "-gencode=arch=compute_90,code=sm_90",
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
