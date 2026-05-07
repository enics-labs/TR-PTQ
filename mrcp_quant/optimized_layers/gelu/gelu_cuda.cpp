#include <torch/extension.h>

void sigmoid_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t bits,
    torch::Tensor out
);

void sigmoid_mixed_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t bits,
    torch::Tensor out
);

void sigmoid_recip_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
);

void sigmoid_alpha_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
);

void sigmoid_alpha_mixed_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
);

void gelu_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
);

void gelu_alpha_table_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor alpha_lut,
    int64_t bits,
    torch::Tensor out
);

void exp_variant_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t variant,
    torch::Tensor out
);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void sigmoid_arith(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t bits,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kInt8, "x must be int8 Q4.4");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(out.dtype() == torch::kUInt8, "out must be uint8");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");
    TORCH_CHECK(bits >= 4 && bits <= 30,
                "bits must be in [4, 30] so ln can be converted to Q4.4");

    sigmoid_arith_cuda(x, exp_lut, bits, out);
}

void sigmoid_mixed_arith(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t bits,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kInt8, "x must be int8 Q4.4");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(out.dtype() == torch::kUInt8, "out must be uint8");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");
    TORCH_CHECK(bits >= 4 && bits <= 30,
                "bits must be in [4, 30] so ln can be converted to Q4.4");

    sigmoid_mixed_arith_cuda(x, exp_lut, bits, out);
}

void sigmoid_recip_arith(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kInt8, "x must be int8 Q4.4");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(out.dtype() == torch::kUInt8, "out must be uint8");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");

    sigmoid_recip_arith_cuda(x, exp_lut, out);
}

void sigmoid_alpha_arith(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(out.dtype() == torch::kUInt8, "out must be uint8");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");
    TORCH_CHECK(bits >= 4 && bits <= 30,
                "bits must be in [4, 30] so ln can be converted to Q4.4");

    sigmoid_alpha_arith_cuda(x, exp_lut, alpha, bits, out);
}

void sigmoid_alpha_mixed_arith(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(out.dtype() == torch::kUInt8, "out must be uint8");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");
    TORCH_CHECK(bits >= 4 && bits <= 30,
                "bits must be in [4, 30] so ln can be converted to Q4.4");

    sigmoid_alpha_mixed_arith_cuda(x, exp_lut, alpha, bits, out);
}

void gelu_arith(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");
    TORCH_CHECK(bits >= 4 && bits <= 30,
                "bits must be in [4, 30] so ln can be converted to Q4.4");

    gelu_arith_cuda(x, exp_lut, alpha, bits, out);
}

void gelu_alpha_table_arith(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor alpha_lut,
    int64_t bits,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(alpha_lut);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(alpha_lut);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(alpha_lut.dtype() == torch::kFloat32, "alpha_lut must be float32");
    TORCH_CHECK(out.dtype() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");
    TORCH_CHECK(alpha_lut.numel() == 10, "alpha_lut must contain exactly 10 values");
    TORCH_CHECK(bits >= 4 && bits <= 30,
                "bits must be in [4, 30] so ln can be converted to Q4.4");

    gelu_alpha_table_arith_cuda(x, exp_lut, alpha_lut, bits, out);
}

void exp_variant_arith(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t variant,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(exp_lut);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(exp_lut);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kInt8, "x must be int8 Q4.4");
    TORCH_CHECK(exp_lut.dtype() == torch::kUInt8, "exp_lut must be uint8");
    TORCH_CHECK(out.dtype() == torch::kUInt8, "out must be uint8");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");
    TORCH_CHECK(variant >= 0 && variant <= 2,
                "variant must be 0, 1, or 2");

    exp_variant_arith_cuda(x, exp_lut, variant, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_arith", &sigmoid_arith,
          "TR-PTQ fused sigmoid approximation for GELU (CUDA)");

    m.def("sigmoid_mixed_arith", &sigmoid_mixed_arith,
          "TR-PTQ mixed exp-variant sigmoid approximation for GELU (CUDA)");

    m.def("sigmoid_recip_arith", &sigmoid_recip_arith,
          "TR-PTQ reciprocal-based sigmoid approximation for GELU (CUDA)");

    m.def("sigmoid_alpha_arith", &sigmoid_alpha_arith,
          "TR-PTQ sigmoid approximation for float alpha*x input (CUDA, uint8 output)");

    m.def("sigmoid_alpha_mixed_arith", &sigmoid_alpha_mixed_arith,
          "TR-PTQ mixed exp-variant sigmoid approximation for float alpha*x input (CUDA, uint8 output)");

    m.def("gelu_arith", &gelu_arith,
          "TR-PTQ GELU approximation using fused sigmoid (CUDA)");

    m.def("gelu_alpha_table_arith", &gelu_alpha_table_arith,
          "TR-PTQ GELU approximation using input-dependent alpha table (CUDA)");

    m.def("exp_variant_arith", &exp_variant_arith,
          "TR-PTQ exp approximation variant for Q4.4 inputs (CUDA)");
}
