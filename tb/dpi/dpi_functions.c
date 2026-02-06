#include <svdpi.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>

// This function calculates the "ideal" ln and converts it to Qx.4 format
// We use 'int' for compatibility with Verilog logic [31:0]
DPI_DLLESPEC int c_ln_reference(int xq_val, int K) {
    if (xq_val <= 0) return 0; // Handle ln(0) or negative as 0 for safety

    // 1. Convert fixed-point input to double (Divide by 2^4)
    double x_double = (double)xq_val / (1 << K);

    // 2. Compute natural log using standard math library
    double ln_result = log(x_double);

    // 3. Convert back to fixed-point (Multiply by 2^4) and round
    int yq_fixed = (int)round(ln_result * (1 << K));

    return yq_fixed;
}


// Convert real -> signed fixed-point Q(M.K)
// Returns int32_t (SV int)
int dpi_real_to_qmk(double real_val, int M, int K)
{
    // Scale factor
    double scale = (double)(1LL << K);

    // Scaled value
    double scaled = real_val * scale;

    // Round to nearest integer
    int64_t fixed = (int64_t) llround(scaled);

    // Total bits: sign + M + K
    int total_bits = 1 + M + K;

    // Saturation limits
    int64_t max_val =  (1LL << (total_bits - 1)) - 1;
    int64_t min_val = -(1LL << (total_bits - 1));

    if (fixed > max_val) fixed = max_val;
    if (fixed < min_val) fixed = min_val;

    return (int32_t) fixed;
}


// Convert signed fixed-point Q(M.K) -> real
double dpi_qmk_to_real(int fixed_val, int K)
{
    // Scale factor
    double scale = (double)(1LL << K);

    // Convert back to real
    return ((double)fixed_val) / scale;
}


// =============================================================
// DPI golden model for reciprocal
// Fixed-point Q(M.K) -> real -> 1/x -> fixed Q1.7
// =============================================================

// xq   : signed fixed-point Q(M.K)
// M,K  : format of input
// Returns signed Q1.7
int dpi_reciprocal_ref(int xq, int M, int K)
{
    // ---------------------------------------------------------
    // Fixed -> real
    // ---------------------------------------------------------
    double x = ((double)xq) / (double)(1LL << K);

    // Guard: division by zero or negative
    if (x <= 0.0) {
        return 0;   // or saturate, depending on your RTL choice
    }

    // ---------------------------------------------------------
    // Reciprocal in real domain
    // ---------------------------------------------------------
    double y = 1.0 / x;

    // ---------------------------------------------------------
    // Real -> Q1.7
    // ---------------------------------------------------------
    double scaled = y * (1 << (K+M));
    int64_t q = (int64_t) llround(scaled);

    // Saturate to signed 8-bit
    if (q > (1 << (M + K)) - 1)  q = ((1 << (M + K)) - 1);
    if (q < 0)     q = 0;

    return (int) q;
}

