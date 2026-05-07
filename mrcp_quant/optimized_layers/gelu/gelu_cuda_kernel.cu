#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cmath>
#include <cstdint>

#define TR_DEFINE_EXP_LUT_CONST
#include "../common/tr_math.cuh"

#define THREADS 256

__device__ __forceinline__
int tr_clamp_u8(int x) {
    return (x < 0) ? 0 : (x > 255 ? 255 : x);
}

__device__ __forceinline__
int tr_clamp_i8_q44(int x) {
    return (x < -128) ? -128 : (x > 127 ? 127 : x);
}

__device__ __forceinline__
int tr_sigmoid_u8_from_q44(int xq, int bits) {
    int x_neg = (xq < 0) ? xq : 0;
    int x_pos_neg = (xq > 0) ? -xq : 0;

    int exp_int = tr_clamp_u8(tr_approx_exp_scalar(x_neg));
    int exp_zero = tr_clamp_u8(tr_approx_exp_scalar(x_pos_neg));

    int ln_sum = tr_new_ln_scalar(exp_int + exp_zero, bits);
    int neg_ln_q44 = -(ln_sum >> (bits - 4));
    neg_ln_q44 = (neg_ln_q44 < -128) ? -128 : (neg_ln_q44 > 0 ? 0 : neg_ln_q44);

    int ln_mul = tr_clamp_u8(tr_approx_exp_scalar(neg_ln_q44));
    return tr_clamp_u8(((ln_mul * exp_int)) >> 8);
}

__device__ __forceinline__
int tr_sigmoid_mixed_u8_from_q44(int xq, int bits) {
    int x_neg = (xq < 0) ? xq : 0;
    int x_pos_neg = (xq > 0) ? -xq : 0;

    int exp_int = tr_approx_exp_scalar_0rd(x_neg);
    int exp_zero = tr_approx_exp_scalar_0rd(x_pos_neg);

    int ln_sum = tr_new_ln_scalar(exp_int + exp_zero, bits);
    int neg_ln_q44 = -(ln_sum >> (bits - 4));
    neg_ln_q44 = (neg_ln_q44 < -128) ? -128 : (neg_ln_q44 > 0 ? 0 : neg_ln_q44);

    int ln_mul = tr_approx_exp_scalar_2rd(neg_ln_q44);
    return tr_clamp_u8((ln_mul * exp_int) >> 8);
}

__device__ __forceinline__
int tr_sigmoid_q16_from_q44(int xq, int bits) {
    int x_neg = (xq < 0) ? xq : 0;
    int x_pos_neg = (xq > 0) ? -xq : 0;

    int exp_int = tr_clamp_u8(tr_approx_exp_scalar(x_neg));
    int exp_zero = tr_clamp_u8(tr_approx_exp_scalar(x_pos_neg));

    int ln_sum = tr_new_ln_scalar(exp_int + exp_zero, bits);
    int neg_ln_q44 = -(ln_sum >> (bits - 4));
    neg_ln_q44 = (neg_ln_q44 < -128) ? -128 : (neg_ln_q44 > 0 ? 0 : neg_ln_q44);

    int ln_mul = tr_clamp_u8(tr_approx_exp_scalar(neg_ln_q44));
    return ln_mul * exp_int;
}

__device__ __forceinline__
int tr_sigmoid_recip_u8_from_q44(int xq) {
    int abs_xq = (xq < 0) ? -xq : xq;
    abs_xq = (abs_xq > 128) ? 128 : abs_xq;
    int neg_abs_xq = -abs_xq;

    int exp_abs = tr_clamp_u8(tr_approx_exp_scalar(neg_abs_xq));
    int denom = 255 + exp_abs;
    int recip = (int)tr_recip_u8_scalar(denom);

    int numer = (xq >= 0) ? 255 : exp_abs;
    return tr_clamp_u8((numer * recip + 128) >> 8);
}

__device__ __forceinline__
uint8_t tr_approx_exp_variant_scalar(int xq, int variant) {
    if (variant == 0) {
        return tr_approx_exp_scalar_0rd(xq);
    }
    if (variant == 1) {
        return tr_approx_exp_scalar(xq);
    }
    return tr_approx_exp_scalar_2rd(xq);
}

__global__
void exp_variant_kernel(
    const int8_t* __restrict__ x,
    uint8_t* __restrict__ out,
    int total,
    int variant
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    out[idx] = tr_approx_exp_variant_scalar((int)x[idx], variant);
}

void exp_variant_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t variant,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();
    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    exp_variant_kernel<<<grid, block>>>(
        x.data_ptr<int8_t>(),
        out.data_ptr<uint8_t>(),
        total,
        (int)variant
    );
}

__global__
void sigmoid_kernel(
    const int8_t* __restrict__ x,
    uint8_t* __restrict__ out,
    int total,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    out[idx] = static_cast<uint8_t>(tr_sigmoid_u8_from_q44((int)x[idx], bits));
}

__global__
void sigmoid_mixed_kernel(
    const int8_t* __restrict__ x,
    uint8_t* __restrict__ out,
    int total,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    out[idx] = static_cast<uint8_t>(tr_sigmoid_mixed_u8_from_q44((int)x[idx], bits));
}

void sigmoid_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t bits,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();
    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    sigmoid_kernel<<<grid, block>>>(
        x.data_ptr<int8_t>(),
        out.data_ptr<uint8_t>(),
        total,
        (int)bits
    );
}

void sigmoid_mixed_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    int64_t bits,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();
    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    sigmoid_mixed_kernel<<<grid, block>>>(
        x.data_ptr<int8_t>(),
        out.data_ptr<uint8_t>(),
        total,
        (int)bits
    );
}

__global__
void sigmoid_recip_kernel(
    const int8_t* __restrict__ x,
    uint8_t* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    out[idx] = static_cast<uint8_t>(tr_sigmoid_recip_u8_from_q44((int)x[idx]));
}

void sigmoid_recip_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();
    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    sigmoid_recip_kernel<<<grid, block>>>(
        x.data_ptr<int8_t>(),
        out.data_ptr<uint8_t>(),
        total
    );
}

__global__
void sigmoid_alpha_kernel(
    const float* __restrict__ x,
    uint8_t* __restrict__ out,
    int total,
    float alpha,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float xv = x[idx];
    int xq = tr_clamp_i8_q44((int)nearbyintf(xv * alpha * 16.0f));
    int sig_u8 = tr_sigmoid_u8_from_q44(xq, bits);
    out[idx] = static_cast<uint8_t>(sig_u8);
}

__global__
void sigmoid_alpha_mixed_kernel(
    const float* __restrict__ x,
    uint8_t* __restrict__ out,
    int total,
    float alpha,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float xv = x[idx];
    int xq = tr_clamp_i8_q44((int)nearbyintf(xv * alpha * 16.0f));
    int sig_u8 = tr_sigmoid_mixed_u8_from_q44(xq, bits);
    out[idx] = static_cast<uint8_t>(sig_u8);
}

void sigmoid_alpha_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();
    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    sigmoid_alpha_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<uint8_t>(),
        total,
        (float)alpha,
        (int)bits
    );
}

void sigmoid_alpha_mixed_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();
    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    sigmoid_alpha_mixed_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<uint8_t>(),
        total,
        (float)alpha,
        (int)bits
    );
}

__global__
void gelu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int total,
    float alpha,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float xv = x[idx];
    int xq = tr_clamp_i8_q44((int)nearbyintf(xv * alpha * 16.0f));
    int sig_q16 = tr_sigmoid_q16_from_q44(xq, bits);
    out[idx] = xv * ((float)sig_q16 / 65536.0f);
}

__global__
void gelu_alpha_table_kernel(
    const float* __restrict__ x,
    const float* __restrict__ alpha_lut,
    float* __restrict__ out,
    int total,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float xv = x[idx];
    float x_clamped = fminf(fmaxf(xv, -5.0f), 5.0f);
    int lut_idx = (int)floorf(x_clamped + 5.0f);
    lut_idx = (lut_idx < 0) ? 0 : (lut_idx > 9 ? 9 : lut_idx);

    float alpha = alpha_lut[lut_idx];
    int xq = tr_clamp_i8_q44((int)nearbyintf(xv * alpha * 16.0f));
    int sig_q16 = tr_sigmoid_q16_from_q44(xq, bits);
    out[idx] = xv * ((float)sig_q16 / 65536.0f);
}

void gelu_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    double alpha,
    int64_t bits,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();
    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        total,
        (float)alpha,
        (int)bits
    );
}

void gelu_alpha_table_arith_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor alpha_lut,
    int64_t bits,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();
    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    gelu_alpha_table_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        alpha_lut.data_ptr<float>(),
        out.data_ptr<float>(),
        total,
        (int)bits
    );
}
