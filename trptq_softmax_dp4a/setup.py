from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='trptq_softmax_dp4a',
    ext_modules=[
        CUDAExtension(
            name='trptq_softmax_dp4a',
            sources=[
                'softmax_cuda.cpp',
                'softmax_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)