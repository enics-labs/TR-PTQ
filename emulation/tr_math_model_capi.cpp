// tr_math_model_capi.cpp — extern "C" wrappers around tr_math_model.hpp for
// ctypes.  Compiled into a shared library (libtr_math_model.so) loaded by
// tools/infra/tr_math_hw.py, so Python calls the SAME RTL-verified functions
// cpu_math_model.cpp uses for verify_block.py -- one implementation, two
// callers, instead of parallel Python/C++ ports that can drift (see
// docs/known_issues.md).
//
// Build (also done automatically by tools/infra/tr_math_hw.py on first import):
//   g++ -O2 -std=c++17 -shared -fPIC \
//       -Istable_code/mrcp_quant/optimized_layers/common -Imock_cuda \
//       tr_math_model_capi.cpp -o libtr_math_model.so

#include "tr_math_model.hpp"

extern "C" {

// x, out: 8 int8 each.
void tr_hw_rmsnorm(const int8_t* x, int8_t* out) {
    rmsnorm_hw_model(x, out);
}

// Batched RMSNorm: n_groups independent 8-element groups in one call (avoids
// one Python<->C round trip per group in a hot loop, e.g. per token per layer).
void tr_hw_rmsnorm_batch(const int8_t* x, int8_t* out, int32_t n_groups) {
    for (int32_t g = 0; g < n_groups; g++)
        rmsnorm_hw_model(x + g * 8, out + g * 8);
}

// x, out: 8 int8 each.
void tr_hw_gelu(const int8_t* x, int8_t* out) {
    gelu_hw_model(x, out);
}

// Batched GELU: n independent elements (no grouping needed -- see
// gelu_hw_scalar's comment) in one call.
void tr_hw_gelu_batch(const int8_t* x, int8_t* out, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] = gelu_hw_scalar(x[i]);
}

// Single-lane requantize: Saturate((acc*mult + bias) >>> shift).
int8_t tr_hw_quant(int64_t acc, int32_t mult, int32_t shift) {
    return quant_model(acc, mult, shift);
}

// Batched requantize: acc[n], shared mult/shift -> out[n].  Avoids one Python
// ctypes call per element for vectorized use.
void tr_hw_quant_batch(const int64_t* acc, int32_t mult, int32_t shift,
                       int8_t* out, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] = quant_model(acc[i], mult, shift);
}

// Matmul tile: A flattened row-major [4][8], b[8], c[4] bias, -> y[4].
void tr_hw_matmul(const int8_t* A_flat, const int8_t* b, const int32_t* c,
                  int32_t mult, int32_t shift, int8_t* y) {
    int8_t A[4][8];
    for (int m = 0; m < 4; m++)
        for (int k = 0; k < 8; k++) A[m][k] = A_flat[m * 8 + k];
    matmul_model(A, b, c, mult, shift, y);
}

// Intermediate RMSNorm passes, exposed for diagnostics/debug printing (e.g.
// tools/scripts/testgen/gen_rmsnorm_test.py) so callers needing the Pass 2/3
// values don't need their own hand-port -- the correctness-critical path is
// tr_hw_rmsnorm() above; these are for display only.
int32_t tr_hw_ln4(int32_t xq) {
    return tr_new_ln_scalar(xq, 4);
}
int32_t tr_hw_exp_backbone(int8_t x) {
    return gelu_exp_q44(x);
}

}  // extern "C"
