#include <torch/extension.h>

void softmax_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor y
);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void softmax_arith_dp4a(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor y
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(y);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(y);

    TORCH_CHECK(x.dtype() == torch::kInt16, "x must be int16 Q4.4");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(y.dtype() == torch::kInt16, "y must be int16");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, cols]");
    TORCH_CHECK(y.sizes() == x.sizes(), "y must have same shape as x");

    softmax_arith_dp4a_cuda(x, exp_lut, y);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_arith_dp4a", &softmax_arith_dp4a, "TR-PTQ Softmax DP4A (CUDA)");
}

#include <torch/extension.h>

void exp_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
);

void exp_arith_dp4a(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
) {
    exp_arith_dp4a_cuda(x, exp_lut, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exp_arith_dp4a", &exp_arith_dp4a,
          "TR-PTQ exponent approximation (CUDA, DP4A)");
}