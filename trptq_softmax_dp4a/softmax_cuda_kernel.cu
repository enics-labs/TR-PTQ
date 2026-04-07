#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cstdint>
#include <climits>

#define THREADS 256
#define A_MIN -8
#define A_MAX 0

// ======================================================
// Constant LUT (shared across all kernels)
// ======================================================
__constant__ uint8_t EXP_LUT_CONST[9];

// ======================================================
// Helpers
// ======================================================
__device__ __forceinline__
int round_q44_to_int(int x_q44) {
    return (x_q44 + 8) >> 4;
}

__device__ __forceinline__
int clamp_anchor(int a) {
    return (a < A_MIN) ? A_MIN : (a > A_MAX ? A_MAX : a);
}

__device__ __forceinline__
uint8_t approx_m_u8(int z_q44, int a) {
    int r = z_q44 - (a << 4);
    int m = 16 + r;
    return static_cast<uint8_t>(m);
}

__device__ __forceinline__
int approx_exp_scalar(int z_q44) {
    int a = clamp_anchor(round_q44_to_int(z_q44));
    int E = static_cast<int>(EXP_LUT_CONST[a - A_MIN]);
    int r = z_q44 - (a << 4);
    int m = 16 + r;
    return (E * m) >> 4;
}

__device__ __forceinline__
int pack_u8x4(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3) {
    return (int)a0 |
           ((int)a1 << 8) |
           ((int)a2 << 16) |
           ((int)a3 << 24);
}

__device__ __forceinline__
int dot_u8x4(int packed_a, int packed_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    return __dp4a(packed_a, packed_b, 0);
#else
    unsigned int a = (unsigned int)packed_a;
    unsigned int b = (unsigned int)packed_b;
    return (int)((a & 0xFF) * (b & 0xFF) +
                 ((a >> 8) & 0xFF) * ((b >> 8) & 0xFF) +
                 ((a >> 16) & 0xFF) * ((b >> 16) & 0xFF) +
                 ((a >> 24) & 0xFF) * ((b >> 24) & 0xFF));
#endif
}

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
    int e = approx_exp_scalar(xq);

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

        int a0 = clamp_anchor(round_q44_to_int(z0));
        int a1 = clamp_anchor(round_q44_to_int(z1));
        int a2 = clamp_anchor(round_q44_to_int(z2));
        int a3 = clamp_anchor(round_q44_to_int(z3));

        int packedE = pack_u8x4(
            EXP_LUT_CONST[a0 - A_MIN],
            EXP_LUT_CONST[a1 - A_MIN],
            EXP_LUT_CONST[a2 - A_MIN],
            EXP_LUT_CONST[a3 - A_MIN]
        );

        int packedM = pack_u8x4(
            approx_m_u8(z0, a0),
            approx_m_u8(z1, a1),
            approx_m_u8(z2, a2),
            approx_m_u8(z3, a3)
        );
        local_sum += dot_u8x4(packedE, packedM);
    }

    // tail
    int tail_start = (cols / 4) * 4;
    for (int i = tail_start + tid; i < cols; i += blockDim.x) {
        int z = (int)row_x[i] - row_max;
        local_sum += (approx_exp_scalar(z) << 4);
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
        local_sum += approx_exp_scalar(z);
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
        int e = approx_exp_scalar(z);
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