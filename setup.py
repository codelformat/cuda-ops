import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()

# 自动收集 csrc/ 下所有源文件，新增算子不需要改这里
sources = sorted(glob.glob("csrc/**/*.cpp", recursive=True) + glob.glob("csrc/**/*.cu", recursive=True))

setup(
    name="cuda_ops",
    version="0.1.0",
    packages=["cuda_ops"],
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="cuda_ops._C",
            sources=sources,
            include_dirs=[str(ROOT / "csrc")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],

            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
