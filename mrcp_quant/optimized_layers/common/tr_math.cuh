#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#define TR_A_MIN -8
#define TR_A_MAX 0
#define TR_Q8_U8_A_MIN -8
#define TR_Q8_Q16_A_MIN -16

#ifdef TR_DEFINE_EXP_LUT_CONST
__constant__ uint8_t EXP_LUT_CONST[9];
#else
extern __constant__ uint8_t EXP_LUT_CONST[9];
#endif

__device__ __forceinline__
int tr_round_q44_to_int(int x_q44) {
    return (x_q44 + 8) >> 4;
}

__device__ __forceinline__
int tr_clamp_anchor(int a) {
    return (a < TR_A_MIN) ? TR_A_MIN : (a > TR_A_MAX ? TR_A_MAX : a);
}

__device__ __forceinline__
int tr_clamp_anchor_range(int a, int a_min, int a_max) {
    return (a < a_min) ? a_min : (a > a_max ? a_max : a);
}

__device__ __forceinline__
int tr_round_q8_to_int(int x_q8) {
    return (x_q8 + 128) >> 8;
}

__device__ __forceinline__
int tr_exp_u8_anchor_q8(int a) {
    switch (a) {
        case -8: return 0;
        case -7: return 0;
        case -6: return 1;
        case -5: return 2;
        case -4: return 5;
        case -3: return 13;
        case -2: return 35;
        case -1: return 94;
        default: return 255;
    }
}

__device__ __forceinline__
int tr_exp_q16_anchor_q8(int a) {
    switch (a) {
        case -16: return 0;
        case -15: return 0;
        case -14: return 0;
        case -13: return 0;
        case -12: return 0;
        case -11: return 1;
        case -10: return 3;
        case -9: return 8;
        case -8: return 22;
        case -7: return 60;
        case -6: return 162;
        case -5: return 442;
        case -4: return 1200;
        case -3: return 3263;
        case -2: return 8869;
        case -1: return 24109;
        default: return 65536;
    }
}

__device__ __forceinline__
uint8_t tr_approx_m_u8(int z_q44, int a) {
    int r = z_q44 - (a << 4);
    int m = 16 + r;
    return static_cast<uint8_t>(m);
}

__device__ __forceinline__
uint8_t tr_approx_exp_scalar_0rd(int z_q44) {
    int a = tr_clamp_anchor(tr_round_q44_to_int(z_q44));
    int E = static_cast<int>(EXP_LUT_CONST[a - TR_A_MIN]);
    return E;
}

__device__ __forceinline__
uint8_t tr_approx_exp_scalar(int z_q44) {
    int a = tr_clamp_anchor(tr_round_q44_to_int(z_q44));
    int E = static_cast<int>(EXP_LUT_CONST[a - TR_A_MIN]);
    int r = z_q44 - (a << 4);
    int m = 16 + r;
    return (E * m) >> 4;
}

__device__ __forceinline__
uint8_t tr_approx_exp_scalar_2rd(int z_q44) {
    int a = tr_clamp_anchor(tr_round_q44_to_int(z_q44));
    int E = static_cast<int>(EXP_LUT_CONST[a - TR_A_MIN]);
    int r = z_q44 - (a << 4);
    int m = 16 + r + ((r*r)>>(4+1));
    return (E * m) >> 4;
}

__device__ __forceinline__
uint8_t tr_approx_exp_q8_u8_scalar(int z_q8) {
    int a = tr_clamp_anchor_range(tr_round_q8_to_int(z_q8), TR_Q8_U8_A_MIN, 0);
    int E = tr_exp_u8_anchor_q8(a);
    int r_q8 = z_q8 - (a << 8);
    int m_q8 = 256 + r_q8;
    int y = (E * m_q8) >> 8;
    y = (y < 0) ? 0 : (y > 255 ? 255 : y);
    return static_cast<uint8_t>(y);
}

__device__ __forceinline__
int tr_approx_exp_q8_q16_i32_scalar(int z_q8) {
    int a = tr_clamp_anchor_range(tr_round_q8_to_int(z_q8), TR_Q8_Q16_A_MIN, 0);
    int E = tr_exp_q16_anchor_q8(a);
    int r_q8 = z_q8 - (a << 8);
    int m_q8 = 256 + r_q8;
    int y = (E * m_q8) >> 8;
    return (y < 0) ? 0 : (y > 65536 ? 65536 : y);
}

__device__ __forceinline__
int tr_pack_u8x4(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3) {
    return (int)a0 |
           ((int)a1 << 8) |
           ((int)a2 << 16) |
           ((int)a3 << 24);
}

__device__ __forceinline__
int tr_dot_u8x4(int packed_a, int packed_b) {
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

__device__ __forceinline__
int tr_floor_log2_positive_i32(int x) {
    return 31 - __clz(static_cast<unsigned int>(x));
}

__device__ __forceinline__
int tr_safe_shift_new_ln(int x, int shift) {
    if (shift >= 0) {
        return (shift >= 31) ? 0 : (x >> shift);
    }

    int left_shift = -shift;
    if (left_shift >= 31) {
        return 0;
    }
    return x << left_shift;
}

__device__ __forceinline__
int tr_new_ln_scalar(int xq, int bits) {
    if (xq <= 0) {
        return 0;
    }

    int aq = tr_floor_log2_positive_i32(xq) - bits;
    int k1 = tr_safe_shift_new_ln(xq, aq);
    int k2 = (aq - 1) << bits;
    int k = k1 + k2;
    return (k >> 1) + (k >> 3) + (k >> 4);
}

__device__ __forceinline__
int tr_new_ln_raw_i32_q8_scalar(int x) {
    if (x <= 0) {
        return 0;
    }

    int a = tr_floor_log2_positive_i32(x);
    int k1 = tr_safe_shift_new_ln(x, a - 8);
    int k2 = (a - 1) << 8;
    int k = k1 + k2;
    return (k >> 1) + (k >> 3) + (k >> 4);
}

__device__ __forceinline__
uint8_t tr_recip_u8_scalar(int x) {
    if (x <= 0) {
        return 0;
    }
    int ln_x_q8 = tr_new_ln_raw_i32_q8_scalar(x);
    return tr_approx_exp_q8_u8_scalar(-ln_x_q8);
}

__device__ __forceinline__
uint16_t tr_recip_u16_scalar(int x) {
    if (x <= 0) {
        return 0;
    }
    int ln_x_q8 = tr_new_ln_raw_i32_q8_scalar(x);
    int y = tr_approx_exp_q8_q16_i32_scalar(-ln_x_q8);
    y = (y < 0) ? 0 : (y > 65535 ? 65535 : y);
    return static_cast<uint16_t>(y);
}

__device__ __forceinline__
int tr_recip_q16_i32_scalar(int x) {
    if (x <= 0) {
        return 0;
    }
    x = (x > (1 << 16)) ? (1 << 16) : x;
    int ln_x_q8 = tr_new_ln_scalar(x, 8);
    return tr_approx_exp_q8_q16_i32_scalar(-ln_x_q8);
}
