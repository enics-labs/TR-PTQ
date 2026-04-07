#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <cstdint>
#include <climits>

#define THREADS 256
#define A_MIN -8
#define A_MAX 0

__constant__ uint8_t EXP_LUT_CONST[9];

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
    return static_cast<int>(a0) |
           (static_cast<int>(a1) << 8) |
           (static_cast<int>(a2) << 16) |
           (static_cast<int>(a3) << 24);
}

__device__ __forceinline__
int dot_u8x4(int packed_a, int packed_b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    return __dp4a(packed_a, packed_b, 0);
#else
    unsigned int a = static_cast<unsigned int>(packed_a);
    unsigned int b = static_cast<unsigned int>(packed_b);
    return static_cast<int>((a & 0xFF) * (b & 0xFF) +
                            ((a >> 8) & 0xFF) * ((b >> 8) & 0xFF) +
                            ((a >> 16) & 0xFF) * ((b >> 16) & 0xFF) +
                            ((a >> 24) & 0xFF) * ((b >> 24) & 0xFF));
#endif
}

template <typename OutT>
__device__ __forceinline__
int clamp_exp_to_output(int e);

template <>
__device__ __forceinline__
int clamp_exp_to_output<int8_t>(int e) {
    if (e < 0) e = 0;
    if (e > 127) e = 127;
    return e;
}

template <>
__device__ __forceinline__
int clamp_exp_to_output<int16_t>(int e) {
    if (e < 0) e = 0;
    if (e > 32767) e = 32767;
    return e;
}

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
        local_max = max(local_max, static_cast<int>(row_x[i]));
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

    int local_sum = 0;

    for (int i = tid * 4; i + 3 < cols; i += blockDim.x * 4) {
        int z0 = static_cast<int>(row_x[i + 0]) - row_max;
        int z1 = static_cast<int>(row_x[i + 1]) - row_max;
        int z2 = static_cast<int>(row_x[i + 2]) - row_max;
        int z3 = static_cast<int>(row_x[i + 3]) - row_max;

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

        local_sum += dot_u8x4(packedE, packedM);
    }

    int tail_start = (cols / 4) * 4;
    for (int i = tail_start + tid; i < cols; i += blockDim.x) {
        int z = static_cast<int>(row_x[i]) - row_max;
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

    int denom = smem[0] >> 4;
    if (denom < 1) denom = 1;

    for (int i = tid; i < cols; i += blockDim.x) {
        int z = static_cast<int>(row_x[i]) - row_max;
        int e = approx_exp_scalar(z);
        int out = (e << 8) / denom;
        row_y[i] = static_cast<int16_t>(out);
    }
}

void softmax_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor y
) {
    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    int rows = static_cast<int>(x.size(0));
    int cols = static_cast<int>(x.size(1));

    dim3 block(THREADS);
    dim3 grid(rows);

    softmax_dp4a_kernel<<<grid, block>>>(
        x.data_ptr<int16_t>(),
        y.data_ptr<int16_t>(),
        rows,
        cols
    );
}

template <typename OutT>
__global__
void exp_kernel(
    const int8_t* __restrict__ x,
    OutT* __restrict__ out,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    int xq = static_cast<int>(x[idx]);
    int e = approx_exp_scalar(xq);
    e = clamp_exp_to_output<OutT>(e);

    out[idx] = static_cast<OutT>(e);
}

void exp_arith_dp4a_cuda(
    torch::Tensor x,
    torch::Tensor exp_lut,
    torch::Tensor out
) {
    int total = static_cast<int>(x.numel());

    cudaMemcpyToSymbol(EXP_LUT_CONST, exp_lut.data_ptr<uint8_t>(), 9);

    dim3 block(THREADS);
    dim3 grid((total + THREADS - 1) / THREADS);

    if (out.dtype() == torch::kInt8) {
        exp_kernel<int8_t><<<grid, block>>>(
            x.data_ptr<int8_t>(),
            out.data_ptr<int8_t>(),
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
