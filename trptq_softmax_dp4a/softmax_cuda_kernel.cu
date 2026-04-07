#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <cstdint>

#define THREADS 256
#define A_MIN -8
#define A_MAX 0

// -----------------------------
// Constant LUT
// -----------------------------
__constant__ uint8_t EXP_LUT_CONST[9];

// -----------------------------
// Helpers
// -----------------------------
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
    return (uint8_t)m;
}

__device__ __forceinline__
int approx_exp_scalar(int z_q44) {
    int a = clamp_anchor(round_q44_to_int(z_q44));
    int r = z_q44 - (a << 4);
    int m = 16 + r;
    int E = EXP_LUT_CONST[a - A_MIN];
    return (E * m) >> 4;
}

__device__ __forceinline__
int pack_u8x4(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3) {
    return ((int)a0) |
           ((int)a1 << 8) |
           ((int)a2 << 16) |
           ((int)a3 << 24);
}

// -----------------------------
// Kernel
// -----------------------------
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

    // =========================
    // 1. MAX (vectorized)
    // =========================
    int local_max = -2147483647;

    int vec_cols = cols >> 2;

    const int4* row_x4 = reinterpret_cast<const int4*>(row_x);

    for (int i = tid; i < vec_cols; i += blockDim.x) {
        int4 v = row_x4[i];

        local_max = max(local_max, v.x);
        local_max = max(local_max, v.y);
        local_max = max(local_max, v.z);
        local_max = max(local_max, v.w);
    }

    // tail
    for (int i = (vec_cols << 2) + tid; i < cols; i += blockDim.x) {
        local_max = max(local_max, (int)row_x[i]);
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = max(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    int row_max = smem[0];

    // =========================
    // 2. DENOM (DP4A + vectorized)
    // =========================
    int local_sum = 0;

    for (int i = tid; i < vec_cols; i += blockDim.x) {
        int4 v = row_x4[i];

        int z0 = v.x - row_max;
        int z1 = v.y - row_max;
        int z2 = v.z - row_max;
        int z3 = v.w - row_max;

        int a0 = clamp_anchor(round_q44_to_int(z0));
        int a1 = clamp_anchor(round_q44_to_int(z1));
        int a2 = clamp_anchor(round_q44_to_int(z2));
        int a3 = clamp_anchor(round_q44_to_int(z3));

        uint8_t E0 = EXP_LUT_CONST[a0 - A_MIN];
        uint8_t E1 = EXP_LUT_CONST[a1 - A_MIN];
        uint8_t E2 = EXP_LUT_CONST[a2 - A_MIN];
        uint8_t E3 = EXP_LUT_CONST[a3 - A_MIN];

        uint8_t M0 = approx_m_u8(z0, a0);
        uint8_t M1 = approx_m_u8(z1, a1);
        uint8_t M2 = approx_m_u8(z2, a2);
        uint8_t M3 = approx_m_u8(z3, a3);

        int packedE = pack_u8x4(E0, E1, E2, E3);
        int packedM = pack_u8x4(M0, M1, M2, M3);

        local_sum = __dp4a(packedE, packedM, local_sum);
    }

    // tail
    for (int i = (vec_cols << 2) + tid; i < cols; i += blockDim.x) {
        int z = (int)row_x[i] - row_max;
        local_sum += approx_exp_scalar(z) << 4;
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    int denom = (smem[0] >> 4);
    if (denom < 1) denom = 1;

    // =========================
    // 3. NORMALIZE (UNROLLED + VECTOR)
    // =========================
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        int4 v = row_x4[i];
        int4 out;

        // ---- UNROLLED ----
        int z0 = v.x - row_max;
        int e0 = approx_exp_scalar(z0);
        out.x = (e0 << 8) / denom;

        int z1 = v.y - row_max;
        int e1 = approx_exp_scalar(z1);
        out.y = (e1 << 8) / denom;

        int z2 = v.z - row_max;
        int e2 = approx_exp_scalar(z2);
        out.z = (e2 << 8) / denom;

        int z3 = v.w - row_max;
        int e3 = approx_exp_scalar(z3);
        out.w = (e3 << 8) / denom;

        reinterpret_cast<int4*>(row_y)[i] = out;
    }

    // tail
    for (int i = (vec_cols << 2) + tid; i < cols; i += blockDim.x) {
        int z = (int)row_x[i] - row_max;
        int e = approx_exp_scalar(z);
        int out = (e << 8) / denom;
        row_y[i] = (int16_t)out;
    }
}

// -----------------------------
// Host launcher
// -----------------------------
void softmax_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor y
) {
    cudaMemcpyToSymbol(
        EXP_LUT_CONST,
        exp_lut.data_ptr<uint8_t>(),
        9
    );

    int rows = x.size(0);
    int cols = x.size(1);

    dim3 block(THREADS);
    dim3 grid(rows);

    softmax_dp4a_kernel<<<grid, block>>>(
        x.data_ptr<int16_t>(),
        y.data_ptr<int16_t>(),
        rows,
        cols
    );
}