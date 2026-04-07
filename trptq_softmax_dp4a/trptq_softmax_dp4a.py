"""Python shim for the CUDA extension.

This ensures PyTorch shared libraries are loaded before importing the compiled
extension module, avoiding libc10/libtorch loader errors in notebook workflows.
"""

import torch  # noqa: F401 - intentionally imported for side effects
from _trptq_softmax_dp4a import *  # noqa: F401,F403
