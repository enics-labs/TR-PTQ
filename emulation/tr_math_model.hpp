#pragma once
// tr_math_model.hpp — single source of truth for the bit-exact hardware math
// models (RMSNorm, GELU, requantize, matmul). Header-only so it can be
// included by BOTH:
//   - cpu_math_model.cpp   (the CLI tool: RTL-verification vector generation
//                            + --eval, driven via verify_block.py / xrun)
//   - tr_math_model_capi.cpp (the extern "C" shared library Python calls
//                              through ctypes — see tools/infra/tr_math_hw.py)
// so there is exactly ONE implementation of each op, not parallel ports that
// can silently drift (see docs/known_issues.md — this is what happened to the
// old Python gelu_int8 approximation).
//
// All functions are RTL-verified via verify_block.py: exp/ln/softmax/swiglu
// were pre-existing; gelu/rmsnorm/quant/matmul were added and confirmed
// against tr_soc_top_int / requantize_engine_int / dot_product_engine.

#include <cstdint>

// CUDA CPU Compilation Overrides (tr_math.cuh is written for nvcc; this makes
// it compile as plain C++ for host-side use).
#ifndef __CUDACC__
    #define __device__
    #define __forceinline__ inline
    #define __constant__ const
    #define __clz(x) __builtin_clz(x)
#endif

static const uint8_t EXP_LUT_CONST[9] = {0, 0, 1, 2, 5, 13, 35, 94, 255};

#include "../emulation/stable_code/mrcp_quant/optimized_layers/common/tr_math.cuh"

// ============================================================================
// GELU helpers — bit-exact replicas of the RTL submodules.
// ============================================================================

// Replicates alpha_stabilizer.sv: computes -|alpha_approx * x| in Q4.4.
inline int8_t alpha_stabilizer_model(int8_t x) {
    uint8_t abs_z = (x < 0) ? (uint8_t)(-(int16_t)x) : (uint8_t)x;
    int16_t x_ext  = (int16_t)x;
    int16_t x_base = (x_ext << 4) + (x_ext << 3);  // x * 24

    int16_t x_mult;
    switch ((abs_z >> 4) & 0x7) {  // abs_z[6:4]
        case 0:  x_mult = x_base + (x_ext << 1) + x_ext; break;  // * 27
        case 1:  x_mult = x_base + (x_ext << 1);          break;  // * 26
        case 2:  x_mult = x_base + x_ext;                 break;  // * 25
        default: x_mult = x_base;                          break;  // * 24
    }

    int8_t x_scaled;
    if      (x_mult >  2032) x_scaled =  127;
    else if (x_mult < -2048) x_scaled = -128;
    else                     x_scaled = (int8_t)((int16_t)x_mult >> 4);  // x_mult[11:4]

    return (x_scaled > 0) ? (int8_t)(-x_scaled) : x_scaled;
}

// Replicates tr_exp_alu (ITER=2) + the GELU is_zero bypass:
//   is_zero → vec_a=128, vec_b=mantisa<<1 → [15:8] = mantisa
//   otherwise → vec_a=LUT[idx], vec_b=mantisa → [15:8] = (LUT*mantisa)>>8
// Returns the Q4.4 exp value (integer, range 0..16). Also reused by
// rmsnorm_hw_model's Pass 3 (same shared VPU backbone — see its comment).
inline int gelu_exp_q44(int8_t x) {
    static const uint8_t EXP_LUT[8] = {94, 35, 13, 5, 2, 1, 0, 0};  // anchors -1..-8

    // round.sv
    int trunc_int      = (int)x >> 4;          // arithmetic shift → integer part
    int frac_round_bit = ((uint8_t)x >> 3) & 1; // bit[3]
    int rounded_mag    = trunc_int + frac_round_bit;
    bool is_zero       = (rounded_mag == 0);

    // tr_exp_alu mantisa (ITER=2, FRAC_W=4)
    int frac_bits   = (uint8_t)x & 0xF;        // x[3:0]
    int is_ceil     = frac_round_bit;
    int first_order = ((!is_ceil) << 4) | frac_bits;  // {~is_ceil, x[3:0]}

    // quadratic_divider K-map (8-bit Q4.4)
    int b3=(frac_bits>>3)&1, b2=(frac_bits>>2)&1, b1=(frac_bits>>1)&1, b0=frac_bits&1;
    int y1 =  b3 & !b2 & !b1 & !b0;
    int y0 = (!b3 & b2 & b1) | (b3 & !b2 & !b1 & b0) | (b3 & !b2 & b1 & !b0);
    int mantisa = (first_order + ((y1 << 1) | y0)) & 0xFF;

    if (is_zero)
        return mantisa;                         // 128*(mantisa<<1) → [15:8] = mantisa

    int lut_idx = (~rounded_mag) & 0x7;
    return (EXP_LUT[lut_idx] * mantisa) >> 8;  // [15:8] of 16-bit product
}

// ============================================================================
// GELU — alpha_stabilizer -> exp -> ln -> exp -> sigma -> multiply, bit-exact
// to the RTL's GELU CMD sequence. Q4.4 fixed point (scale 1/16) throughout —
// confirmed against tr_soc_ctrl_int.sv's "vecmul_scale_mode = 2'b01; // Slice
// [11:4] for Q4.4 output". Purely element-wise (no cross-lane coupling, unlike
// RMSNorm's sum-of-squares) — the RTL processes 8 lanes in parallel because
// that's the VPU's vector width, not because the math needs grouping, so the
// scalar primitive below is the true unit and the 8-wide wrapper is just a
// loop over it.
// ============================================================================
inline int8_t gelu_hw_scalar(int8_t x) {
    int8_t alpha = alpha_stabilizer_model(x);
    int    E     = gelu_exp_q44(alpha);
    int    pre_ln  = E + 16;
    int    ln_yq8  = tr_new_ln_scalar(pre_ln << 4, 8);
    if (ln_yq8 >  2047) ln_yq8 =  2047;
    if (ln_yq8 < -2048) ln_yq8 = -2048;
    int8_t ln_out  = (int8_t)((ln_yq8 + 8) >> 4);
    int8_t neg_ln  = (int8_t)(-(int16_t)ln_out);
    int    recip   = gelu_exp_q44(neg_ln);
    int    sigma   = (x < 0) ? (16 - recip) : recip;
    int    product = (int)x * (int8_t)sigma;
    return (int8_t)(product >> 4);
}

inline void gelu_hw_model(const int8_t x[8], int8_t out[8]) {
    for (int i = 0; i < 8; i++) out[i] = gelu_hw_scalar(x[i]);
}

// ============================================================================
// RMSNorm — bit-exact replica of the tr_soc_ctrl_int RM_P1..RM_P4 FSM sequence.
// Composed from the SAME primitives above rather than a third independent
// port: Pass 2 (ln) -> tr_new_ln_scalar(xq,4); Pass 3 (exp) -> gelu_exp_q44.
// ============================================================================
static const int RMSNORM_CONST_LN_SQRT_N = 17;   // 0.5*ln(8) in Q4.4, N=8

inline void rmsnorm_hw_model(const int8_t x[8], int8_t out[8]) {
    // Pass 1: S = sum(x[i]^2), INT32
    int32_t S = 0;
    for (int i = 0; i < 8; i++) S += (int32_t)x[i] * (int32_t)x[i];

    // Pass 2: log_s = ln(S[19:4]) in Q4.4
    int s_shifted = (S >> 4) & 0xFFFF;
    int log_s = tr_new_ln_scalar(s_shifted, 4);
    if (log_s > 127) log_s = 127; if (log_s < -128) log_s = -128;
    int reg_scalar_log = -(log_s >> 1);
    if (reg_scalar_log > 127) reg_scalar_log = 127;
    if (reg_scalar_log < -128) reg_scalar_log = -128;

    // Pass 3: InvRMS = exp(reg_scalar_log + CONST_LN_SQRT_N)
    int ctrl_scalar = reg_scalar_log + RMSNORM_CONST_LN_SQRT_N;
    if (ctrl_scalar > 127) ctrl_scalar = 127; if (ctrl_scalar < -128) ctrl_scalar = -128;
    int inv_rms = gelu_exp_q44((int8_t)ctrl_scalar);   // 0..255

    // Pass 4: out[i] = (x[i] * InvRMS) >> 4, SV assignment truncates (wraps),
    // does not saturate — matches the RTL's W_VEC' cast.
    for (int i = 0; i < 8; i++) {
        int product = (int)x[i] * inv_rms;
        int shifted = product >> 4;
        out[i] = (int8_t)shifted;   // truncating cast (wraps on overflow)
    }
}

// ============================================================================
// Quant / requantize — bit-exact replica of requantize_engine_int.sv:
//   Saturate((acc*mult + bias) >>> shift, [-128,127]),  bias = shift>0 ? 1<<(shift-1) : 0
// ============================================================================
inline int8_t quant_model(int64_t acc, int32_t mult, int shift) {
    int64_t v = acc * (int64_t)mult;
    if (shift > 0) v += (int64_t)1 << (shift - 1);
    v = (shift > 0) ? (v >> shift) : v;
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

// ============================================================================
// Matmul — bit-exact replica of dot_product_engine + requantize_engine_int
// chained (the tr_soc_top_int single-tile path): y[m] = quant(c[m] +
// sum_k(A[m][k]*b[k]), mult, shift).  M=4, N=8 — matches the production tile
// size (tr_matmul_ctrl streams this tile-by-tile with clear_acc accumulation,
// which is mathematically the same full accumulate-then-quantize result).
// ============================================================================
inline void matmul_model(const int8_t A[4][8], const int8_t b[8],
                         const int32_t c[4], int32_t mult, int shift,
                         int8_t y[4]) {
    for (int m = 0; m < 4; m++) {
        int32_t acc = c[m];
        for (int k = 0; k < 8; k++) acc += (int32_t)A[m][k] * (int32_t)b[k];
        y[m] = quant_model(acc, mult, shift);
    }
}
