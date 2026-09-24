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
// Returns the Q4.4 exp value (integer, range 0..16).
//
// gelu_exp_q44_raw takes the FULL-WIDTH value (no pre-narrowing to int8):
// round.sv's ">>4 then check bit 3" truncation/rounding is applied directly
// to whatever precision the caller has, matching the RTL, where (per
// tr_nonlinear_vpu.sv's MUX 4/5 in GL_P2 and post_ln_modifier's negate mode)
// the round+LUT stage consumes the backbone's wide LN output directly --
// there is no intermediate narrow-to-int8 rounding step in hardware between
// negate(ln(...)) and the second exp lookup. gelu_exp_q44(int8_t) is kept as
// a thin wrapper for callers that already have a genuinely narrow INT8 value
// (GL_P1's alpha_stabilizer output; RMSNorm Pass 3's ctrl_scalar, both
// already clamped to [-128,127] before the call).
// Raw (UNSHIFTED) tr_exp_alu product: E(Q0.8) x mantisa(Q4.4), scaled by
// 4096 -- the same round.sv + quadratic_divider.sv K-map bit-trick used by
// every exp lookup in the RTL (GELU, RMSNorm, Softmax alike), factored out
// so each caller applies its own datapath's shift/saturation instead of
// three copies of this LUT/rounding logic silently drifting apart (see the
// file header note on gelu_int8 -- this is exactly that trap).
inline int tr_exp_alu_product_raw(int x) {
    static const uint8_t EXP_LUT[8] = {94, 35, 13, 5, 2, 1, 0, 0};  // anchors -1..-8

    // round.sv
    int trunc_int      = x >> 4;          // arithmetic shift → integer part
    int frac_round_bit = (x >> 3) & 1;    // bit[3]
    int rounded_mag    = trunc_int + frac_round_bit;
    bool is_zero       = (rounded_mag == 0);

    // tr_exp_alu mantisa (ITER=2, FRAC_W=4)
    int frac_bits   = x & 0xF;                 // x[3:0]
    int is_ceil     = frac_round_bit;
    int first_order = ((!is_ceil) << 4) | frac_bits;  // {~is_ceil, x[3:0]}

    // quadratic_divider K-map (8-bit Q4.4)
    int b3=(frac_bits>>3)&1, b2=(frac_bits>>2)&1, b1=(frac_bits>>1)&1, b0=frac_bits&1;
    int y1 =  b3 & !b2 & !b1 & !b0;
    int y0 = (!b3 & b2 & b1) | (b3 & !b2 & !b1 & b0) | (b3 & !b2 & b1 & !b0);
    int mantisa = (first_order + ((y1 << 1) | y0)) & 0xFF;

    if (is_zero)
        return 128 * (mantisa << 1);   // mac_in_a/vecmul_a's is_zero->128 doubling trick

    int lut_idx = (~rounded_mag) & 0x7;
    return EXP_LUT[lut_idx] * mantisa;
}

// GELU's exp lookup consumes the raw product at [15:8] (Q4.4, scale 16):
// is_zero -> 128*(mantisa<<1) = 256*mantisa, >>8 = mantisa (unchanged from
// before this was factored out); non-zero -> (E*mantisa)>>8, also unchanged.
inline int gelu_exp_q44_raw(int x) { return tr_exp_alu_product_raw(x) >> 8; }

inline int gelu_exp_q44(int8_t x) { return gelu_exp_q44_raw((int)x); }

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
    // bits=4 (matching FRAC_W=4 -- the same precision RMSNorm's verified
    // tr_new_ln_scalar(s_shifted, 4) call uses), input NOT pre-shifted by
    // <<4 -- confirmed bit-exact against tr_soc_top_int's actual bb_log
    // register (512/512 match); the previous bits=8 + extra <<4/>>4 shuffle
    // was never independently RTL-verified (it was reverse-engineered from
    // tr_gelu.sv, the dead-code module -- see tr_gelu_int_tb.sv) and simply
    // used the wrong LN precision.
    int    ln_q44  = tr_new_ln_scalar(pre_ln, 4);
    int8_t neg_ln  = (int8_t)(-(int16_t)(int8_t)ln_q44);
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

// PROPOSED FIX (not real hardware -- see docs/iscas_paper_support write-up,
// Part 5, for the analysis). Root cause: ctrl_scalar goes POSITIVE for small
// Sigma x^2 (small S -> less-negative/positive log_s -> reg_scalar_log can
// exceed -CONST_LN_SQRT_N), and gelu_exp_q44's shared backbone
// (tr_exp_alu_product_raw) is a DECAY-only evaluator -- it was designed for
// alpha_stabilizer's always-non-positive output and has no mechanism to
// evaluate a growing exponential; for the pathological positive-ctrl_scalar
// case its rounded_mag/lut_idx bit-trick simply breaks down (see the
// write-up for the exact mechanism), collapsing inv_rms to near 0 instead
// of growing.
//
// Fix: sign-guard ctrl_scalar (force it non-positive before calling the
// SAME existing decay-side backbone, mirroring alpha_stabilizer's own
// "forced negative absolute value" step -- cheap: one comparator, one
// two's-complement negate, one mux), then for the ORIGINAL-positive case,
// recover the needed growing value via a reciprocal: exp(+t) = 1/exp(-t).
// Concretely: E = gelu_exp_q44(-|ctrl_scalar|) is already computed
// correctly by the EXISTING decay-only backbone (no new exp/ln evaluation
// needed); inv_rms = 256/E is then a SINGLE integer division (256 = 16*16,
// converting Q4.4 E back through the same Q4.4 output convention). This is
// NOT a reuse of tr_recip_u8_scalar (that primitive uses different, wider
// Q8 ln/exp primitives not otherwise used by this Q4.4 backbone, and is
// unused/unsynthesized anywhere in this codebase's RTL today) -- it's a
// small dedicated divider, new arithmetic scoped to the RMSNorm wrapper,
// not a control-only fix and not a shared-backbone change.
inline void rmsnorm_hw_model_signguard_fixed(const int8_t x[8], int8_t out[8]) {
    int32_t S = 0;
    for (int i = 0; i < 8; i++) S += (int32_t)x[i] * (int32_t)x[i];

    int s_shifted = (S >> 4) & 0xFFFF;
    int log_s = tr_new_ln_scalar(s_shifted, 4);
    if (log_s > 127) log_s = 127; if (log_s < -128) log_s = -128;
    int reg_scalar_log = -(log_s >> 1);
    if (reg_scalar_log > 127) reg_scalar_log = 127;
    if (reg_scalar_log < -128) reg_scalar_log = -128;

    int ctrl_scalar = reg_scalar_log + RMSNORM_CONST_LN_SQRT_N;
    if (ctrl_scalar > 127) ctrl_scalar = 127; if (ctrl_scalar < -128) ctrl_scalar = -128;

    bool was_positive = ctrl_scalar > 0;
    int8_t ctrl_stab = (int8_t)(was_positive ? -ctrl_scalar : ctrl_scalar);
    int E = gelu_exp_q44(ctrl_stab);           // exp(-|ctrl_scalar|), always correct (decay side)
    int inv_rms;
    if (!was_positive) {
        inv_rms = E;                            // ctrl_scalar<=0: unchanged from the original model
    } else {
        int E_safe = (E < 1) ? 1 : E;            // guard divide-by-zero
        inv_rms = (256 + E_safe / 2) / E_safe;   // round-to-nearest integer division
    }
    if (inv_rms > 255) inv_rms = 255;
    if (inv_rms < 0) inv_rms = 0;

    for (int i = 0; i < 8; i++) {
        int product = (int)x[i] * inv_rms;
        int shifted = product >> 4;
        out[i] = (int8_t)shifted;
    }
}

// ============================================================================
// Softmax — bit-exact replica of the tr_soc_top_int PRODUCTION path (the
// SM_P1..SM_P4 FSM in tr_soc_ctrl_int.sv driving tr_nonlinear_vpu), NOT the
// standalone tr_softmax.sv module (that one is a differently-architected,
// separate RTL block -- explicit reciprocal-via-exp instead of this FSM's
// log-sum-exp trick, and its own is_zero->255 anchor instead of this
// pipeline's is_zero->128-doubling trick -- and isn't instantiated by
// tr_soc_top_int.sv at all; see tr_softmax_int_tb.sv / cpu_model
// "softmax_dead_module" for that one instead).
//
// max -> Sum(exp(xi-max)) -> ln(Sum) -> exp(xi-max-ln(Sum)), no divider.
// Output is Q0.8 UNSIGNED (0..255, saturating): softmax values are always in
// [0,1], so the whole byte is spent as fraction instead of wasting 4 bits on
// an integer range that's never used (matches tr_nonlinear_vpu.sv's
// vecmul_scale_mode==2'b11 path).
// ============================================================================
inline int8_t softmax_sat8_sub(int a, int b) {
    int diff = a - b;
    if (diff < -128) diff = -128;
    if (diff > 127) diff = 127;
    return (int8_t)diff;
}

// z_out (optional, may be NULL): the tile's own internal log-partition value
// (SM_P4's sub_val2 = max_x + ln(S), int8, same domain as x). Exposed ONLY
// for modeling a controller-level streaming/multi-tile composition on top of
// this fixed-N=8 primitive -- softmax_hw_model() itself is unchanged RTL
// behavior; this is an additive, backward-compatible parameter (existing
// callers passing NULL, including the original signature below, are
// bit-identical to before). See docs/vit_inference_completion_plan.md /
// tools/scripts/vpu_precision_archive/hw_function_accuracy.py's streaming_softmax_sweep() for why: exp(z_out) is
// exactly this tile's own absolute exp-sum (Sum_i exp(x_i)), so combining
// per-tile z_out values via a running logsumexp, using the SAME shared
// exp/ln backbone already used here, correctly composes softmax across
// tiles wider than N=8 -- this is NOT implemented in any existing RTL,
// firmware, or testbench (confirmed by inspection: tr_soc_ctrl_int.sv's
// SM_P1..SM_P4 FSM and tr_softmax_int_tb.sv both only ever process one
// independent N=8 tile per invocation, and the real firmware path,
// tr_softmax_row() in tr_tensor.c, is pure CPU code that never touches this
// VPU primitive at all).
inline void softmax_hw_model_ex(const int8_t x[8], uint8_t out[8], int8_t* z_out) {
    int8_t max_x = x[0];
    for (int i = 1; i < 8; i++) if (x[i] > max_x) max_x = x[i];

    // SM_P1/P2: S = sum_i exp(x_i - max), exact 32-bit accumulate (mirrors
    // mac_array_engine's UU multiply-accumulate -- no rounding loss here).
    int32_t S = 0;
    for (int i = 0; i < 8; i++) {
        int8_t d = softmax_sat8_sub(x[i], max_x);
        S += tr_exp_alu_product_raw(d);
    }

    // SM_P3: ln(S) via the [23:8] bit-slice into tr_ln_alu (bits=4, matching
    // FRAC_W=4 -- same primitive GELU/RMSNorm already verify), saturated to
    // int8 (tr_ln_alu.sv itself saturates its OUT_WIDTH-bit output).
    int slice16 = (int)((uint32_t)S >> 8) & 0xFFFF;
    int ln_S = tr_new_ln_scalar(slice16, 4);
    if (ln_S > 127) ln_S = 127;
    if (ln_S < -128) ln_S = -128;

    // ctrl_scalar_sub_val = max + ln(S), saturating (tr_soc_ctrl_int.sv's
    // SM_P4 state widens to 9 bits and clamps before truncating back to
    // int8 -- a plain 8-bit add here used to silently wrap on large
    // max+ln(S), corrupting the whole row's exponent argument).
    int sub_val2_wide = (int)max_x + (int)ln_S;
    if (sub_val2_wide > 127) sub_val2_wide = 127;
    if (sub_val2_wide < -128) sub_val2_wide = -128;
    int8_t sub_val2 = (int8_t)sub_val2_wide;
    if (z_out) *z_out = sub_val2;

    // SM_P4: softmax_i = exp(x_i - max - ln(S)), Q0.8 UNSIGNED, saturating.
    for (int i = 0; i < 8; i++) {
        int8_t d2 = softmax_sat8_sub(x[i], sub_val2);
        int shifted = tr_exp_alu_product_raw(d2) >> 4;
        out[i] = (uint8_t)((shifted > 255) ? 255 : shifted);
    }
}

inline void softmax_hw_model(const int8_t x[8], uint8_t out[8]) {
    softmax_hw_model_ex(x, out, nullptr);
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
