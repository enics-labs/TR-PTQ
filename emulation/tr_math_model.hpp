#pragma once
// tr_math_model.hpp — single source of truth for the bit-exact hardware math
// models (RMSNorm, GELU, requantize, matmul). Header-only so it can be
// included by BOTH:
//   - cpu_math_model.cpp   (the CLI tool: RTL-verification vector generation
//                            + --eval, driven via verify_block.py / xrun)
//   - tr_math_model_capi.cpp (the extern "C" shared library Python calls
//                              through ctypes — see tools/infra/tr_math_hw.py)
// so there is exactly ONE implementation of each op, not parallel ports that
// can silently drift (see docs/platform/known_issues.md — this is what happened to the
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

#include <cmath>
#include <vector>

// ============================================================================
// tr_ln_alu golden model (piecewise-linear ln, replaces the shipped
// (k>>1)+(k>>3)+(k>>4) datapath). Bit-exact to rtl_soc/.../tr_ln_alu.sv and
// to the software reference tools/models/vpu_ops.py::_ln_core(ln_mode="pwl").
//   fk  = the `bits` mantissa bits below the leading one, seg = top log2(segments) bits of fk
//   aq  = msb - bits
//   yq  = (aq*L*2^bits + A[seg]*fk + B[seg]*2^bits + 2^(c-1)) >> c      (arithmetic shift)
// L=round(ln2*2^c); A/B = round(2^c * least-squares line of ln(1+g) over segment `seg`
// of [0,1), fitted on 2000 grid points per segment -- same fit as vpu_ops._segments).
// Result saturated to out_width signed bits. xq<=0 -> 0.
// NOTE: tr_new_ln_scalar (tr_math.cuh) is the OLD datapath and is left untouched for the
// CUDA kernels in stable_code; the RTL no longer matches it. Every RTL tr_ln_alu instance
// (tr_backbone_wrapper and the rtl_baseline modules) uses the module defaults SEGMENTS=2,
// COEF_BITS=8. tr_hw_ln() is the module-level unit (rtl_baseline modules instantiate it directly);
// tr_vpu_ln() is what the tr_soc production path sees through tr_backbone_wrapper (shared B=6 unit).
// ============================================================================
struct TrLnPwlCoefs { int L; std::vector<int> A, B; };

inline const TrLnPwlCoefs& tr_ln_pwl_coefs(int segments, int coef_bits) {
    static std::vector<std::pair<int, TrLnPwlCoefs>> cache;
    int key = segments * 64 + coef_bits;
    for (auto& e : cache) if (e.first == key) return e.second;
    TrLnPwlCoefs c;
    c.L = (int)std::nearbyint(std::log(2.0) * (1 << coef_bits));
    for (int sgi = 0; sgi < segments; sgi++) {
        const int NPT = 2000;
        double lo = (double)sgi / segments, hi = (double)(sgi + 1) / segments;
        long double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < NPT; i++) {
            long double g = lo + (hi - lo) * i / NPT, y = std::log1p((double)g);
            sx += g; sy += y; sxx += g * g; sxy += g * y;
        }
        long double slope = (NPT * sxy - sx * sy) / (NPT * sxx - sx * sx);
        long double icept = (sy - slope * sx) / NPT;
        c.A.push_back((int)std::nearbyint((double)(slope * (1 << coef_bits))));
        c.B.push_back((int)std::nearbyint((double)(icept * (1 << coef_bits))));
    }
    cache.emplace_back(key, c);
    return cache.back().second;
}

inline int tr_pwl_ln_scalar(int xq, int bits, int segments, int coef_bits, int out_width);

// ln as instantiated in the RTL today: default tr_ln_alu parameters (2 segments, 8-bit constants).
inline int tr_hw_ln(int xq, int bits, int out_width = 8) { return tr_pwl_ln_scalar(xq, bits, 2, 8, out_width); }

// The ln as seen through tr_backbone_wrapper (tr_soc production path): ONE shared unit at
// BITS = FRAC_W + 2 = 6 (2 segments, 8-bit constants, 10-bit output). Operators with FRAC_W=4
// fractional bits feed x<<2 (pre_scaled=false); RMSNorm feeds its mean already at 6 fractional
// bits (pre_scaled=true). The Q.6 result is rounded back to Q4.4 and saturated to int8.
// Input is treated as unsigned (callers mask to the 16-bit backbone width) and clamped to the unit's 13-bit input.
inline int tr_vpu_ln(int xq, bool pre_scaled = false) {
    int xin = pre_scaled ? xq : (xq << 2);
    if (xin > (1 << 13) - 1) xin = (1 << 13) - 1;     // LN_IN_W = 13: larger inputs are clamped (unreachable: max reachable is 4096)
    int y = tr_pwl_ln_scalar(xin, 6, 2, 8, 10);
    int r = (y + 2) >> 2;
    return r > 127 ? 127 : (r < -128 ? -128 : r);
}

inline int tr_pwl_ln_scalar(int xq, int bits, int segments, int coef_bits, int out_width) {
    if (xq <= 0) return 0;
    const TrLnPwlCoefs& c = tr_ln_pwl_coefs(segments, coef_bits);
    int msb = 31 - __builtin_clz((unsigned)xq);
    int aq  = msb - bits;
    // mantissa fraction: the `bits` bits below the leading one (zero-padded when msb < bits)
    int fk = (msb >= bits) ? (int)(((unsigned)xq >> (msb - bits)) & ((1u << bits) - 1u))
                           : (int)(((unsigned)xq << (bits - msb)) & ((1u << bits) - 1u));
    int lg  = (segments > 1) ? __builtin_ctz((unsigned)segments) : 0;
    int seg = (lg > 0) ? (fk >> (bits - lg)) : 0;
    long long acc = ((long long)aq * c.L << bits) + (long long)c.A[seg] * fk
                  + ((long long)c.B[seg] << bits) + (1LL << (coef_bits - 1));
    long long yq = acc >> coef_bits;                       // arithmetic shift (floor), as the RTL's >>>
    long long hi = (1LL << (out_width - 1)) - 1, lo = -(1LL << (out_width - 1));
    return (int)(yq > hi ? hi : (yq < lo ? lo : yq));
}

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
// silu_hw_scalar is the x*sigma(x) stage shared bit-for-bit by GELU's GL_P1-
// P3 crossbar sequence and (per tr_swiglu.sv's own header comment, "Passes
// 0-2 are identical to tr_gelu") tr_swiglu's passes 0-2 -- this is ONE
// implementation, not two copies that could drift. Note sigma() here is
// sigmoid(alpha_stabilizer(x)), NOT sigmoid(x): alpha_stabilizer is a
// piecewise GELU-sigmoid-slope approximation (coefficients 27/26/25/24
// sixteenths, tuned for GELU's ~1.702 constant), reused unmodified by
// tr_swiglu.sv for what should be plain SiLU's sigmoid(x) (alpha=1). This is
// a real, measured systematic bias (continuous-math RMSE ~0.095, max ~0.16
// over |x|<6) baked into the existing, already-synthesized tr_swiglu.sv --
// not something introduced here. See swiglu_exact()/silu_exact() in
// tools/scripts/vpu_precision/tr_vpu_model.py for the true-SiLU fp32
// reference this is measured against.
inline int8_t silu_hw_scalar(int8_t x) {
    int8_t alpha = alpha_stabilizer_model(x);
    int    E     = gelu_exp_q44(alpha);
    int    pre_ln  = E + 16;
    // bits=4 (matching FRAC_W=4 -- the same precision RMSNorm's verified
    // tr_hw_ln(s_shifted, 4) call uses), input NOT pre-shifted by
    // <<4 -- confirmed bit-exact against tr_soc_top_int's actual bb_log
    // register (512/512 match); the previous bits=8 + extra <<4/>>4 shuffle
    // was never independently RTL-verified (it was reverse-engineered from
    // tr_gelu.sv, the dead-code module -- see tr_gelu_int_tb.sv) and simply
    // used the wrong LN precision. NOTE: this BITS=4 crossbar precision is
    // deliberately lower than tr_swiglu.sv's own dedicated BITS=8 ln step
    // (see tr_swiglu.sv lines ~53-62) -- 31/256 int8 inputs disagree by up
    // to 2 LSB between the two. swiglu_hw_model() below is bit-exact to the
    // SHARED CROSSBAR (GELU/future SwiGLU-opcode) precision, deliberately
    // NOT to tr_swiglu.sv's standalone module -- that comparison is the
    // point of the exercise, not a bug; see docs/llama/overview.md.
    int    ln_q44  = tr_vpu_ln(pre_ln);
    int8_t neg_ln  = (int8_t)(-(int16_t)(int8_t)ln_q44);
    int    recip   = gelu_exp_q44(neg_ln);
    int    sigma   = (x < 0) ? (16 - recip) : recip;
    int    product = (int)x * (int8_t)sigma;
    return (int8_t)(product >> 4);
}

inline int8_t gelu_hw_scalar(int8_t x) { return silu_hw_scalar(x); }

inline void gelu_hw_model(const int8_t x[8], int8_t out[8]) {
    for (int i = 0; i < 8; i++) out[i] = gelu_hw_scalar(x[i]);
}

// ============================================================================
// SwiGLU (shared-crossbar precision) -- silu_hw_scalar (above) followed by
// one SS multiply, [11:4] slice -- matches tr_soc_ctrl_int's planned
// SW_P1..P3 (silu, mirroring GL_P1..GL_P3) + SW_P4 (gate-multiply, mirroring
// RM_P4's SS-mode structure) opcode sequence bit-for-bit. Deliberately NOT
// bit-exact to tr_swiglu.sv's standalone module (see silu_hw_scalar's
// comment) -- this is the shared-datapath data point in the
// shared-crossbar-vs-dedicated-module comparison.
// ============================================================================
inline int8_t swiglu_hw_scalar(int8_t x, int8_t g) {
    int8_t silu = silu_hw_scalar(x);
    int product = (int)silu * (int)g;
    return (int8_t)(product >> 4);
}

inline void swiglu_hw_model(const int8_t x[8], const int8_t g[8], int8_t out[8]) {
    for (int i = 0; i < 8; i++) out[i] = swiglu_hw_scalar(x[i], g[i]);
}

// ============================================================================
// SwiGLU "best-fit" (flat alpha=1 + saturating gate-multiply) -- HISTORICAL REFERENCE ONLY. This was the model of
// the first SwiGLU opcode pair (CMD=5/6 via alpha_stabilizer flat_mode and vecmul_scale_mode 3'b100). Both RTL
// features have been removed (replaced by the two-pass sigma-ROM flow, sw_* below); the functions are kept because
// tools/models/llama32_1b.py and tools/scripts/llama32_1b/ablate.py use them as the "as shipped" SwiGLU baseline of
// the accuracy study (RTL history: git tag tr-vpu_rtl-ISCAS-20261013 and the commits after it). It therefore keeps the
// OLD shipped ln (tr_new_ln_scalar), not the PWL ln of the current RTL.
// ============================================================================
inline int8_t alpha_stabilizer_model_flat(int8_t x) {
    // coef=16 (1.0 in Q4.4): x*16, >>4 == x unchanged, still saturate+negate
    // the same way the real RTL's flat_mode path does.
    int16_t x_mult = (int16_t)x << 4;
    int8_t x_scaled;
    if      (x_mult >  2032) x_scaled =  127;
    else if (x_mult < -2048) x_scaled = -128;
    else                     x_scaled = (int8_t)(x_mult >> 4);
    return (x_scaled > 0) ? (int8_t)(-x_scaled) : x_scaled;
}

inline int8_t silu_hw_scalar_bestfit(int8_t x) {
    int8_t alpha = alpha_stabilizer_model_flat(x);
    int    E     = gelu_exp_q44(alpha);
    int    pre_ln  = E + 16;
    int    ln_q44  = tr_new_ln_scalar(pre_ln, 4);   // the OLD shipped ln (tr_math.cuh): this model is the pre-Llama baseline
    int8_t neg_ln  = (int8_t)(-(int16_t)(int8_t)ln_q44);
    int    recip   = gelu_exp_q44(neg_ln);
    int    sigma   = (x < 0) ? (16 - recip) : recip;
    int    product = (int)x * (int8_t)sigma;
    return (int8_t)(product >> 4);
}

inline int8_t swiglu_hw_scalar_bestfit(int8_t x, int8_t g) {
    int8_t silu = silu_hw_scalar_bestfit(x);
    int product = (int)silu * (int)g;
    int shifted = product >> 4;
    if (shifted > 127) return 127;
    if (shifted < -128) return -128;
    return (int8_t)shifted;
}

inline void swiglu_hw_model_bestfit(const int8_t x[8], const int8_t g[8], int8_t out[8]) {
    for (int i = 0; i < 8; i++) out[i] = swiglu_hw_scalar_bestfit(x[i], g[i]);
}

// tr_swiglu.sv's OWN standalone precision (BITS=8 ln step, with explicit
// widen-by-<<4 / round-narrow-by->>4 -- see tr_swiglu.sv lines ~53-62),
// bit-exact to its RTL (confirmed via verify_block.py swiglu_baseline,
// 2048/2048). Mirrors cpu_math_model.cpp's "swiglu_baseline" branch
// exactly. This is a SEPARATE,
// higher-precision sigma(alpha_stabilizer(x)) evaluation than
// silu_hw_scalar's BITS=4 shared-crossbar path above -- the two disagree on
// 31/256 int8 inputs (max 2 LSB); see silu_hw_scalar's comment.
inline int8_t silu_hw_scalar_dedicated(int8_t x) {
    int8_t alpha = alpha_stabilizer_model(x);
    int    E     = gelu_exp_q44(alpha);
    int    pre_ln = E + 16;
    int    ln_yq8 = tr_hw_ln(pre_ln << 4, 8, 12);
    if (ln_yq8 >  2047) ln_yq8 =  2047;
    if (ln_yq8 < -2048) ln_yq8 = -2048;
    int8_t ln_out_q44 = (int8_t)((ln_yq8 + 8) >> 4);
    int8_t neg_ln = (int8_t)(-(int16_t)ln_out_q44);
    int    recip  = gelu_exp_q44(neg_ln);
    int    sigma  = (x < 0) ? (16 - recip) : recip;
    int    product = (int)x * (int8_t)sigma;
    return (int8_t)(product >> 4);
}

inline int8_t swiglu_hw_scalar_dedicated(int8_t x, int8_t g) {
    int8_t silu = silu_hw_scalar_dedicated(x);
    int product = (int)silu * (int)g;
    return (int8_t)(product >> 4);
}

inline void swiglu_hw_model_dedicated(const int8_t x[8], const int8_t g[8], int8_t out[8]) {
    for (int i = 0; i < 8; i++) out[i] = swiglu_hw_scalar_dedicated(x[i], g[i]);
}

// ============================================================================
// RMSNorm — bit-exact replica of the tr_soc_ctrl_int RM_P1..RM_P4 FSM sequence.
// Composed from the SAME primitives above rather than a third independent
// port: Pass 2 (ln) -> tr_hw_ln(xq,4); Pass 3 (exp) -> gelu_exp_q44.
// ============================================================================
static const int RMSNORM_CONST_LN_SQRT_N = 17;   // 0.5*ln(8) in Q4.4, N=8

// ----------------------------------------------------------------------------
// SwiGLU, Llama-width (ffn_h = silu(gate) * up over rows of N = 8*groups), host-sequenced in 8-lane groups.
// Software spec: tools/models/vpu_ops.py::swiglu (RECOMMENDED, sw_gain="dynamic", two-pass realization).
//   sw_silu_group : silu_s[i] = sat12((x[i]*SIGMA[x[i]] + 2) >> 2)   (sigma ROM, x = gate code, Q4.4)
//   sw_max_group  : running max of |silu_s[i]*up[i]| over the row         (pass 1, per group)
//   sw_gain       : e = clip(floor(log2(127*2^10 / (4*max))), -8, 10)      (once per row; consumes the max)
//   sw_apply_group: y[i] = sat8(round_shift(4*silu_s[i]*up[i], 10 - e))    (pass 2, per group)
// SIGMA[x+128] = sigmoid(x/16) in Q.6 as produced by the alpha->exp->+1->ln->neg->exp chain at FRAC_W=6 with the
// unified ln (flat alpha, x<0: 64-recip); a 256x6-bit ROM in the VPU replaces the three chain passes.
// ----------------------------------------------------------------------------
static const uint8_t SW_SIGMA[256] = {
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
     2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  4,  4,  4,  4,  5,  5,  6,  6,  6,  7,  7,  7,
     7,  8,  8,  9, 10, 10, 11, 11, 11, 12, 13, 14, 14, 14, 15, 17,
    17, 17, 19, 20, 21, 21, 22, 22, 24, 24, 27, 28, 29, 30, 31, 32,
    31, 32, 33, 34, 35, 36, 37, 40, 40, 42, 42, 43, 43, 44, 45, 47,
    47, 47, 49, 50, 50, 50, 51, 52, 53, 53, 53, 54, 54, 55, 56, 56,
    57, 57, 57, 57, 58, 58, 58, 59, 59, 60, 60, 60, 60, 61, 61, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62,
    62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
};
static const int SW_FRAC_W = 6;
static const int SW_TOTAL  = SW_FRAC_W + 4;   // prod = silu * up has FRAC_W+4 fractional bits
static const int SW_DROP   = 2;               // silu_w (14 bit) -> silu_s (12 bit)
static const int SW_E_MAX  = SW_TOTAL;        // gain cap: shift = SW_TOTAL - e >= 0 (no left shift); real rows have e in [-3, 2]

inline int32_t sw_silu(int8_t x) {
    int32_t w = (int32_t)x * (int32_t)SW_SIGMA[(int)x + 128];
    int32_t s = (w + (1 << (SW_DROP - 1))) >> SW_DROP;
    return s > 2047 ? 2047 : (s < -2048 ? -2048 : s);
}

inline int32_t sw_abs_prod(int32_t silu_s, int8_t up) {   // |silu_s * up| (unscaled by the <<SW_DROP)
    int32_t p = silu_s * (int32_t)up;
    return p < 0 ? -p : p;
}

inline int sw_gain(int32_t max_abs_p) {
    int64_t mx = (int64_t)max_abs_p << SW_DROP;
    if (mx < 1) mx = 1;
    const int64_t R = 127LL << SW_TOTAL;
    int e = -8;
    for (int k = -8; k <= SW_E_MAX; k++) {                 // largest k with mx * 2^k <= R
        bool ok = k >= 0 ? ((mx << k) <= R) : (mx <= (R << -k));
        if (ok) e = k;
    }
    return e;
}

inline int8_t sw_apply(int32_t silu_s, int8_t up, int e) {
    int64_t prod = ((int64_t)silu_s * (int64_t)up) << SW_DROP;
    int sh = SW_TOTAL - e;                                 // >= 0 because e <= SW_E_MAX
    int64_t y = sh > 0 ? ((prod + (1LL << (sh - 1))) >> sh) : prod;
    return (int8_t)(y > 127 ? 127 : (y < -128 ? -128 : y));
}

// ----------------------------------------------------------------------------
// RMSNorm, Llama-width (N = 8 * tiles) -- software spec is
// tools/models/vpu_ops.py::rmsnorm (RECOMMENDED knobs: rms_k=2, rms_inv_bits=4, rounded shifts,
// W_MAC=16, unified ln at BITS=FRAC_W+k). Decomposed exactly as the hardware will run it, one
// opcode per step (host sequences the tiles):
//   rms_acc_tile  : one MAC beat, S += sum(x[i]^2) over 8 lanes   (first tile clears S)
//   rms_scale     : S -> mean-of-squares -> ln -> -0.5x -> exp -> wide inv_rms (once per row)
//   rms_apply_tile: out[i] = sat8((x[i]*inv_rms + half) >> (FRAC_W+INV_BITS))   (per tile)
// The old single-tile rmsnorm_hw_model below is kept only as the golden for the rtl_baseline
// tr_rmsnorm.sv module.
// ----------------------------------------------------------------------------
static const int RMS_FRAC_W   = 4;
static const int RMS_K        = 2;     // extra mean bits kept through the ln
static const int RMS_INV_BITS = 4;     // extra inv_rms fractional bits (exp raw >> (8 - INV_BITS))
static const int RMS_W_MAC    = 16;    // saturation width of the mean fed to the ln
static const int RMS_INV_W    = 12;    // inv_rms register width (unsigned); growth branch capped at 2^12-1

inline int32_t rms_acc_tile(const int8_t x[8], int32_t acc, bool clear) {
    int32_t s = clear ? 0 : acc;
    for (int i = 0; i < 8; i++) s += (int32_t)x[i] * (int32_t)x[i];
    return s;
}

// returns the unsigned wide inv_rms (Q.(FRAC_W+INV_BITS), RMS_INV_W bits)
inline uint32_t rms_scale(int32_t S, int log2n) {
    const int shift = RMS_FRAC_W + log2n - RMS_K;                      // >= 1 for N >= 8
    int64_t mk = ((int64_t)S + (1LL << (shift - 1))) >> shift;         // rounded mean, FRAC_W+k fractional bits
    if (mk > (1LL << RMS_W_MAC) - 1) mk = (1LL << RMS_W_MAC) - 1;      // W_MAC saturation
    int log_s  = tr_vpu_ln((int)mk, true);                             // shared ln, mk already at BITS=FRAC_W+k
    int half   = -((log_s + 1) >> 1);                                  // -0.5 ln(mean), rounded
    int ctrl   = half > 127 ? 127 : (half < -128 ? -128 : half);
    bool growth = ctrl > 0;
    int guarded = growth ? -ctrl : ctrl;                               // exp backbone is decay-only
    int ew = tr_exp_alu_product_raw(guarded) >> (8 - RMS_INV_BITS);
    if (!growth) return (uint32_t)ew;
    int d = ew < 1 ? 1 : ew;
    uint32_t inv = ((1u << (2 * RMS_FRAC_W + 2 * RMS_INV_BITS)) + (uint32_t)(d >> 1)) / (uint32_t)d;
    return inv > (1u << RMS_INV_W) - 1 ? (1u << RMS_INV_W) - 1 : inv;   // cap: unreachable for max|x|=127 rows
}

inline void rms_apply_tile(const int8_t x[8], uint32_t inv, int8_t out[8]) {
    const int sh = RMS_FRAC_W + RMS_INV_BITS;
    for (int i = 0; i < 8; i++) {
        int64_t y = ((int64_t)x[i] * (int64_t)inv + (1LL << (sh - 1))) >> sh;
        out[i] = (int8_t)(y > 127 ? 127 : (y < -128 ? -128 : y));
    }
}

inline void rmsnorm_hw_model(const int8_t x[8], int8_t out[8]) {
    // Pass 1: S = sum(x[i]^2), INT32
    int32_t S = 0;
    for (int i = 0; i < 8; i++) S += (int32_t)x[i] * (int32_t)x[i];

    // Pass 2: log_s = ln(S[19:4]) in Q4.4
    int s_shifted = (S >> 4) & 0xFFFF;
    int log_s = tr_hw_ln(s_shifted, 4);
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
    int log_s = tr_hw_ln(s_shifted, 4);
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
// bit-identical to before). See docs/vit/accuracy-findings.md /
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
    int ln_S = tr_vpu_ln(slice16);
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
