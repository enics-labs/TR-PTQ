#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <climits>
#include <type_traits>

#define TR_DEFINE_EXP_LUT_CONST
#include "../common/tr_math.cuh"

#define THREADS 256

// ======================================================
// EXP KERNEL (int8 → uint8/int16)
// ======================================================
template <typename OutT>
__global__
void exp_kernel(
    const int8_t* __restrict__ x,
    OutT* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int xq = (int)x[idx];
    int e = tr_approx_exp_scalar(xq);

    if constexpr (std::is_same<OutT, uint8_t>::value) {
        e = (e < 0) ? 0 : (e > 255 ? 255 : e);
    } else {
        e = (e < 0) ? 0 : (e > 32767 ? 32767 : e);
    }

    out[idx] = (OutT)e;
}

void exp_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int total = (int)x.numel();

    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    if (out.dtype() == torch::kUInt8) {
        exp_kernel<uint8_t><<<grid, block>>>(
            x.data_ptr<int8_t>(),
            out.data_ptr<uint8_t>(),
            total
        );
    } else {
        exp_kernel<int16_t><<<grid, block>>>(
            x.data_ptr<int8_t>(),
            out.data_ptr<int16_t>(),
            total
        );
    }
}

// ======================================================
// LN KERNEL: mirrors mrcp_quant.quant_utils.new_ln
// ======================================================
__global__
void ln_kernel(
    const int32_t* __restrict__ xq,
    int32_t* __restrict__ out,
    int total,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    out[idx] = tr_new_ln_scalar((int)xq[idx], bits);
}

void ln_arith_cuda(
    torch::Tensor xq,
    int64_t bits,
    torch::Tensor out
) {
    int total = (int)xq.numel();

    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    ln_kernel<<<grid, block>>>(
        xq.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(),
        total,
        (int)bits
    );
}

// ======================================================
// RECIPROCAL KERNELS: tr-exp(-tr-ln(x))
// ======================================================
__global__
void recip_u8_kernel(
    const int32_t* __restrict__ x,
    uint8_t* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    out[idx] = tr_recip_u8_scalar((int)x[idx]);
}

void recip_u8_cuda(
    torch::Tensor x,
    torch::Tensor out
) {
    int total = (int)x.numel();

    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    recip_u8_kernel<<<grid, block>>>(
        x.data_ptr<int32_t>(),
        out.data_ptr<uint8_t>(),
        total
    );
}

__global__
void recip_u16_kernel(
    const int32_t* __restrict__ x,
    uint16_t* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    out[idx] = tr_recip_u16_scalar((int)x[idx]);
}

void recip_u16_cuda(
    torch::Tensor x,
    torch::Tensor out
) {
    int total = (int)x.numel();

    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    recip_u16_kernel<<<grid, block>>>(
        x.data_ptr<int32_t>(),
        out.data_ptr<uint16_t>(),
        total
    );
}

__global__
void recip_q16_i32_kernel(
    const int32_t* __restrict__ x,
    int32_t* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    out[idx] = tr_recip_q16_i32_scalar((int)x[idx]);
}

void recip_q16_i32_cuda(
    torch::Tensor x,
    torch::Tensor out
) {
    int total = (int)x.numel();

    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    recip_q16_i32_kernel<<<grid, block>>>(
        x.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(),
        total
    );
}

// ======================================================
// SIGMOID KERNEL: fused TR-exp + TR-ln + TR-exp
// ======================================================
__global__
void sigmoid_kernel(
    const int8_t* __restrict__ x,
    uint8_t* __restrict__ out,
    int total,
    int bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int xq = (int)x[idx];
    int x_neg = (xq < 0) ? xq : 0;
    int x_pos_neg = (xq > 0) ? -xq : 0;

    int exp_int = tr_approx_exp_scalar(x_neg);
    int exp_zero = tr_approx_exp_scalar(x_pos_neg);
    exp_int = (exp_int < 0) ? 0 : (exp_int > 255 ? 255 : exp_int);
    exp_zero = (exp_zero < 0) ? 0 : (exp_zero > 255 ? 255 : exp_zero);

    int ln_sum = tr_new_ln_scalar(exp_int + exp_zero, bits);
    int neg_ln_q44 = -(ln_sum >> (bits - 4));
    neg_ln_q44 = (neg_ln_q44 < -128) ? -128 : (neg_ln_q44 > 0 ? 0 : neg_ln_q44);

    int ln_mul = tr_approx_exp_scalar(neg_ln_q44);
    ln_mul = (ln_mul < 0) ? 0 : (ln_mul > 255 ? 255 : ln_mul);

    int sigmoid = ((ln_mul * exp_int) + 127) / 255;
    sigmoid = (sigmoid < 0) ? 0 : (sigmoid > 255 ? 255 : sigmoid);
    out[idx] = static_cast<uint8_t>(sigmoid);
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

// ======================================================
// SOFTMAX SUM KERNEL (NEW, int8 path, fused exp+sum)
// ======================================================
__global__
void softmax_sum_kernel(
    const int8_t* __restrict__ x,
    int32_t* __restrict__ out,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const int8_t* row_x = x + row * cols;

    __shared__ int smem[THREADS];

    // -------------------------
    // 1. MAX
    // -------------------------
    int local_max = INT_MIN;
    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = max(local_max, (int)row_x[i]);
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = max(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    int row_max = smem[0];

    // -------------------------
    // 2. SUM (DP4A)
    // -------------------------
    int local_sum = 0;

    for (int i = tid * 4; i + 3 < cols; i += blockDim.x * 4) {
        int z0 = (int)row_x[i + 0] - row_max;
        int z1 = (int)row_x[i + 1] - row_max;
        int z2 = (int)row_x[i + 2] - row_max;
        int z3 = (int)row_x[i + 3] - row_max;

        int a0 = tr_clamp_anchor(tr_round_q44_to_int(z0));
        int a1 = tr_clamp_anchor(tr_round_q44_to_int(z1));
        int a2 = tr_clamp_anchor(tr_round_q44_to_int(z2));
        int a3 = tr_clamp_anchor(tr_round_q44_to_int(z3));

        int packedE = tr_pack_u8x4(
            EXP_LUT_CONST[a0 - TR_A_MIN],
            EXP_LUT_CONST[a1 - TR_A_MIN],
            EXP_LUT_CONST[a2 - TR_A_MIN],
            EXP_LUT_CONST[a3 - TR_A_MIN]
        );

        int packedM = tr_pack_u8x4(
            tr_approx_m_u8(z0, a0),
            tr_approx_m_u8(z1, a1),
            tr_approx_m_u8(z2, a2),
            tr_approx_m_u8(z3, a3)
        );
        local_sum += tr_dot_u8x4(packedE, packedM);
    }

    // tail
    int tail_start = (cols / 4) * 4;
    for (int i = tail_start + tid; i < cols; i += blockDim.x) {
        int z = (int)row_x[i] - row_max;
        local_sum += (tr_approx_exp_scalar(z) << 4);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int denom = smem[0] >> 4;
        if (denom < 1) denom = 1;
        out[row] = denom;
    }
}

void softmax_sum_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int rows = (int)x.size(0);
    int cols = (int)x.size(1);

    dim3 block(THREADS);
    dim3 grid(rows);

    softmax_sum_kernel<<<grid, block>>>(
        x.data_ptr<int8_t>(),
        out.data_ptr<int32_t>(),
        rows,
        cols
    );
}

// ======================================================
// ORIGINAL SOFTMAX (kept unchanged)
// ======================================================
__global__
void softmax_dp4a_kernel(
    const int16_t* __restrict__ x,
    int16_t* __restrict__ y,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    const int16_t* row_x = x + row * cols;
    int16_t* row_y = y + row * cols;

    __shared__ int smem[THREADS];

    int local_max = INT_MIN;

    for (int i = tid; i < cols; i += blockDim.x) {
        local_max = max(local_max, (int)row_x[i]);
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = max(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    int row_max = smem[0];

    int local_sum = 0;

    for (int i = tid; i < cols; i += blockDim.x) {
        int z = (int)row_x[i] - row_max;
        local_sum += tr_approx_exp_scalar(z);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    int denom = smem[0];
    if (denom < 1) denom = 1;

    for (int i = tid; i < cols; i += blockDim.x) {
        int z = (int)row_x[i] - row_max;
        int e = tr_approx_exp_scalar(z);
        row_y[i] = (int16_t)((e << 8) / denom);
    }
}

void softmax_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor y
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int rows = (int)x.size(0);
    int cols = (int)x.size(1);

    dim3 block(THREADS);
    dim3 grid(rows);

    softmax_dp4a_kernel<<<grid, block>>>(
        x.data_ptr<int16_t>(),
        y.data_ptr<int16_t>(),
        rows,
        cols
    );
}
