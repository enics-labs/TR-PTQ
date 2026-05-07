import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TORCH_LIB_DIR = os.path.join(os.path.dirname(torch.__file__), 'lib')


class CleanBuildExtension(BuildExtension):
    """Build extension while removing stale legacy module artifacts."""

    def run(self):
        here = Path(__file__).resolve().parent
        for stale in here.glob('trptq_softmax_dp4a*.so'):
            try:
                stale.unlink()
            except OSError:
                pass
        super().run()


extra_link_args = []
if os.name != 'nt':
    extra_link_args.append(f'-Wl,-rpath,{TORCH_LIB_DIR}')

setup(
    name='trptq_softmax_dp4a',
    py_modules=['trptq_softmax_dp4a'],
    ext_modules=[
        CUDAExtension(
            name='_trptq_softmax_dp4a',
            sources=[
                'softmax_cuda.cpp',
                'softmax_cuda_kernel.cu',
            ],
            library_dirs=[TORCH_LIB_DIR],
            runtime_library_dirs=[TORCH_LIB_DIR] if os.name != 'nt' else [],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={'build_ext': CleanBuildExtension},
)
