#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// CUDA CPU Compilation Overrides
#ifndef __CUDACC__
    #define __device__ 
    #define __forceinline__ inline
    #define __constant__ const
    #define __clz(x) __builtin_clz(x)
#endif

// LUT Constant
const uint8_t EXP_LUT_CONST[9] = {0, 0, 1, 2, 5, 13, 35, 94, 255};

// CUDA Math header
#include "../emulation/stable_code/mrcp_quant/optimized_layers/common/tr_math.cuh"

// GELU helpers — bit-exact replicas of the RTL submodules
// ---------------------------------------------------------

// Replicates alpha_stabilizer.sv: computes -|alpha_approx * x| in Q4.4.
static int8_t alpha_stabilizer_model(int8_t x) {
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
static int gelu_exp_q44(int8_t x) {
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "[ERROR] Must provide a block mode (e.g., './cpu_model exp' or './cpu_model ln')" << std::endl;
        return -1;
    }

    std::string mode = argv[1];
    std::ofstream vec_file("inputs.txt");
    std::ofstream exp_file("expected.txt");

    if (!vec_file.is_open() || !exp_file.is_open()) {
        std::cerr << "[ERROR] Could not open IO files!" << std::endl;
        return -1;
    }

    if (mode == "exp") {
        vec_file << "129\n"; // Header for valid negative domain + zero
        for (int i = -128; i <= 0; i++) {
            int z_q44 = i;
            int expected_y = tr_approx_exp_scalar_2rd(z_q44);
            vec_file << z_q44 << "\n";
            exp_file << z_q44 << " " << expected_y << "\n";
        }
        std::cout << "[C++ MODEL] Generated EXP test vectors." << std::endl;
    } 
    else if (mode == "ln") {
        vec_file << "65536\n"; // Header for exhaustive 16-bit domain
        for (int i = 0; i <= 65535; i++) {
            int xq = i;
            int expected_y = tr_new_ln_scalar(xq, 4);
            
            // Replicate the hardware's 8-bit physical saturation limits
            if (expected_y > 127) expected_y = 127;
            if (expected_y < -128) expected_y = -128;

            vec_file << xq << "\n";
            exp_file << xq << " " << expected_y << "\n";
        }
        std::cout << "[C++ MODEL] Generated LN test vectors." << std::endl;
    } 
    else if (mode == "softmax") {
        vec_file << "256\n"; 
        srand(1337); 
        
        for (int i = 0; i < 256; i++) {
            int x[8];
            int row_max = -128; // Simulating INT_MIN for Q4.4 8-bit limits
            
            // 1. Generate Row and Find Max (Mirrors CUDA Pass 1)
            for (int j = 0; j < 8; j++) {
                x[j] = (rand() % 256) - 128;
                if (x[j] > row_max) row_max = x[j];
                vec_file << x[j] << (j == 7 ? "" : " ");
            }
            vec_file << "\n";

            // 2. Compute Sum of Exponents (Mirrors CUDA Pass 2)
            // Uses 2nd-order Taylor approximation matching HDL tr_exp_alu ITER=2.
            int local_sum = 0;
            for (int j = 0; j < 8; j++) {
                int z = x[j] - row_max;
                local_sum += tr_approx_exp_scalar_2rd(z);
            }

            int denom = local_sum;
            if (denom < 1) denom = 1;

            // 3. Compute Final Output with Division (Mirrors CUDA Pass 3)
            for (int j = 0; j < 8; j++) {
                int z = x[j] - row_max;
                int e = tr_approx_exp_scalar_2rd(z);
                
                // CUDA logic (Q0.8):
                int y_q08 = (e << 8) / denom; 
                
                // Scale down to Q4.4 to match hardware output format
                // A logical shift right by 4 converts Q0.8 (256 scale) to Q4.4 (16 scale)
                int16_t y = (int16_t)(y_q08 >> 4);
                
                exp_file << y << (j == 7 ? "" : " ");
            }
            exp_file << "\n";
        }
        std::cout << "[C++ MODEL] Generated SOFTMAX vectors using 2nd-order exp + exact division." << std::endl;
    }
    else if (mode == "swiglu") {
        const int N_LANES = 8;
        const int N_VECS  = 256;
        vec_file << N_VECS << "\n";
        srand(1337);

        for (int i = 0; i < N_VECS; i++) {
            int x[N_LANES], g[N_LANES];

            for (int j = 0; j < N_LANES; j++) x[j] = (rand() % 256) - 128;
            for (int j = 0; j < N_LANES; j++) g[j] = (rand() % 256) - 128;

            // Write x[0..7] g[0..7] on one line
            for (int j = 0; j < N_LANES; j++) vec_file << x[j] << " ";
            for (int j = 0; j < N_LANES; j++) vec_file << g[j] << (j == N_LANES - 1 ? "" : " ");
            vec_file << "\n";

            for (int j = 0; j < N_LANES; j++) {
                int8_t xq = (int8_t)x[j];
                int8_t gq = (int8_t)g[j];

                // Pass 0: E = exp(-alpha|x|)
                int8_t alpha_q44 = alpha_stabilizer_model(xq);
                int    E_q44     = gelu_exp_q44(alpha_q44);

                // Pass 1: recip = 1/(1+E) via exp(-ln(1+E))
                int    pre_ln_q44  = E_q44 + 16;
                int    ln_yq8      = tr_new_ln_scalar(pre_ln_q44 << 4, 8);
                if (ln_yq8 >  2047) ln_yq8 =  2047;
                if (ln_yq8 < -2048) ln_yq8 = -2048;
                int8_t ln_out_q44  = (int8_t)((ln_yq8 + 8) >> 4);
                int8_t neg_ln_q44  = (int8_t)(-(int16_t)ln_out_q44);
                int    recip_q44   = gelu_exp_q44(neg_ln_q44);

                // Pass 2: silu = x * sigma(x)  [mirrors SU multiply → [11:4]]
                int sigma_q44  = (xq < 0) ? (16 - recip_q44) : recip_q44;
                int silu_prod  = (int32_t)(int8_t)xq * (int32_t)(uint8_t)sigma_q44;
                int8_t silu_q44 = (int8_t)(silu_prod >> 4);

                // Pass 3: y = silu * g  [SS multiply → [11:4]]
                int32_t final_prod = (int32_t)(int8_t)silu_q44 * (int32_t)(int8_t)gq;
                int8_t  y_q44      = (int8_t)(final_prod >> 4);

                exp_file << (int)y_q44 << (j == N_LANES - 1 ? "" : " ");
            }
            exp_file << "\n";
        }
        std::cout << "[C++ MODEL] Generated SWIGLU vectors." << std::endl;
    }
    else if (mode == "gelu") {
        const int N_LANES = 8;
        const int N_VECS  = 256;
        vec_file << N_VECS << "\n";
        srand(1337);

        for (int i = 0; i < N_VECS; i++) {
            int x[N_LANES];
            for (int j = 0; j < N_LANES; j++) {
                x[j] = (rand() % 256) - 128;
                vec_file << x[j] << (j == N_LANES - 1 ? "" : " ");
            }
            vec_file << "\n";

            for (int j = 0; j < N_LANES; j++) {
                int8_t xq = (int8_t)x[j];

                // Pass 0: E = exp(-alpha|x|)  [alpha_stabilizer → tr_exp_alu]
                int8_t alpha_q44 = alpha_stabilizer_model(xq);
                int    E_q44     = gelu_exp_q44(alpha_q44);

                // Pass 1: recip = 1/(1+E)  via  exp(-ln(1+E))
                // pre_ln_modifier adds 1.0 (16 in Q4.4), giving (E+16) in Q4.4.
                // tr_ln_alu (WIDTH=12, BITS=8) expects Q4.8 → zero-pad by <<4.
                int    pre_ln_q44  = E_q44 + 16;                  // (E+1) in Q4.4
                int    ln_yq8      = tr_new_ln_scalar(pre_ln_q44 << 4, 8); // Q4.8
                // Saturate to 12-bit signed
                if (ln_yq8 >  2047) ln_yq8 =  2047;
                if (ln_yq8 < -2048) ln_yq8 = -2048;
                // Shift back Q4.8 → Q4.4 with rounding (factor 8)
                int8_t ln_out_q44  = (int8_t)((ln_yq8 + 8) >> 4);
                // post_ln_modifier mode 01: negate
                int8_t neg_ln_q44  = (int8_t)(-(int16_t)ln_out_q44);
                int    recip_q44   = gelu_exp_q44(neg_ln_q44);

                // Pass 2: y = x * sigma(x)
                // symmetry_modifier: sigma(x) = (x<0) ? 1-recip : recip
                int sigma_q44 = (xq < 0) ? (16 - recip_q44) : recip_q44;
                // vec_mul SS mode (op_mode=0): signed × signed, [11:4] extract
                int product   = (int)xq * (int)(int8_t)sigma_q44;
                int8_t y_q44  = (int8_t)(product >> 4);

                exp_file << (int)y_q44 << (j == N_LANES - 1 ? "" : " ");
            }
            exp_file << "\n";
        }
        std::cout << "[C++ MODEL] Generated GELU vectors." << std::endl;
    }
    else {
        std::cerr << "[ERROR] Unknown mode: " << mode << std::endl;
        return -1;
    }

    vec_file.close();
    exp_file.close();
    return 0;
}