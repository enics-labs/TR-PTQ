#include <torch/extension.h>

// ---------------- CUDA declarations ----------------
void softmax_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor y
);

void exp_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
);

void ln_arith_cuda(
    torch::Tensor xq,
    int64_t bits,
    torch::Tensor out
);

void recip_u8_cuda(
    torch::Tensor x,
    torch::Tensor out
);

void recip_u16_cuda(
    torch::Tensor x,
    torch::Tensor out
);

void recip_q16_i32_cuda(
    torch::Tensor x,
    torch::Tensor out
);

void sigmoid_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t bits,
    torch::Tensor out
);

// NEW
void softmax_sum_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
);

// ---------------- checks ----------------
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------- softmax (existing) ----------------
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
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");

    softmax_arith_dp4a_cuda(x, exp_lut, y);
}

// ---------------- exponent (existing) ----------------
void exp_arith_dp4a(
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
    TORCH_CHECK(out.dtype() == torch::kUInt8 || out.dtype() == torch::kInt16,
                "out must be uint8 or int16");

    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(exp_lut.numel() == 9, "exp_lut must contain exactly 9 values");

    exp_arith_dp4a_cuda(x, exp_lut, out);
}

// ---------------- ln approximation for GELU sigmoid denominator ----------------
void ln_arith(
    torch::Tensor xq,
    int64_t bits,
    torch::Tensor out
) {
    CHECK_CUDA(xq);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(xq);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(xq.dtype() == torch::kInt32, "xq must be int32");
    TORCH_CHECK(out.dtype() == torch::kInt32, "out must be int32");
    TORCH_CHECK(out.numel() == xq.numel(), "out must match xq size");
    TORCH_CHECK(bits >= 0 && bits <= 30, "bits must be in [0, 30]");

    ln_arith_cuda(xq, bits, out);
}

// ---------------- reciprocal approximation with tr-ln + Q8 tr-exp ----------------
void recip_u8(
    torch::Tensor x,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kInt32, "x must be int32");
    TORCH_CHECK(out.dtype() == torch::kUInt8, "out must be uint8");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have same shape as x");

    recip_u8_cuda(x, out);
}

void recip_u16(
    torch::Tensor x,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kInt32, "x must be int32");
    TORCH_CHECK(out.dtype() == torch::kUInt16, "out must be uint16");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have same shape as x");

    recip_u16_cuda(x, out);
}

void recip_q16_i32(
    torch::Tensor x,
    torch::Tensor out
) {
    CHECK_CUDA(x);
    CHECK_CUDA(out);

    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(out);

    TORCH_CHECK(x.dtype() == torch::kInt32, "x must be int32");
    TORCH_CHECK(out.dtype() == torch::kInt32, "out must be int32");
    TORCH_CHECK(out.numel() == x.numel(), "out must match x size");
    TORCH_CHECK(out.sizes() == x.sizes(), "out must have same shape as x");

    recip_q16_i32_cuda(x, out);
}

// ---------------- sigmoid approximation for GELU ----------------
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

// ---------------- NEW: softmax sum ----------------
void softmax_sum_dp4a(
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
    TORCH_CHECK(out.dtype() == torch::kInt32, "out must be int32");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [rows, cols]");
    TORCH_CHECK(out.dim() == 2 || out.dim() == 1,
                "out must be [rows] or [rows,1]");

    TORCH_CHECK(out.size(0) == x.size(0),
                "out must match number of rows");

    TORCH_CHECK(exp_lut.numel() == 9,
                "exp_lut must contain exactly 9 values");

    softmax_sum_dp4a_cuda(x, exp_lut, out);
}

// ---------------- module ----------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_arith_dp4a", &softmax_arith_dp4a,
          "TR-PTQ Softmax DP4A (CUDA)");

    m.def("exp_arith_dp4a", &exp_arith_dp4a,
          "TR-PTQ exponent approximation (CUDA, DP4A)");

    m.def("ln_arith", &ln_arith,
          "TR-PTQ new_ln approximation (CUDA)");

    m.def("recip_u8", &recip_u8,
          "TR-PTQ reciprocal approximation using Q8 tr-ln/tr-exp (CUDA, uint8 output)");

    m.def("recip_u16", &recip_u16,
          "TR-PTQ reciprocal approximation using Q8 tr-ln/tr-exp (CUDA, uint16 Q16 output)");

    m.def("recip_q16_i32", &recip_q16_i32,
          "TR-PTQ reciprocal approximation using Q8 tr-ln/tr-exp (CUDA, int32 Q16 output)");

    m.def("sigmoid_arith", &sigmoid_arith,
          "TR-PTQ sigmoid approximation for GELU (CUDA)");

    m.def("softmax_sum_dp4a", &softmax_sum_dp4a,
          "TR-PTQ softmax denominator (sum) (CUDA, DP4A)");
}
