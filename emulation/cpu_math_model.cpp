#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Single source of truth for all the bit-exact hardware math models
// (alpha_stabilizer_model, gelu_exp_q44, gelu_hw_model, rmsnorm_hw_model,
// quant_model, matmul_model) and the tr_math.cuh exp/ln primitives — shared
// with the ctypes-callable library (tr_math_model_capi.cpp /
// tools/infra/tr_math_hw.py) so there is exactly one implementation of each op.
#include "tr_math_model.hpp"

// ── --eval FILE support: run a model function over externally-supplied      //
// values (e.g. captured firmware activations) instead of self-generated     //
// random vectors.  Reuses the exact same functions the self-test path uses. //
static bool run_eval_mode(const std::string& mode, const std::string& in_path,
                          const std::string& out_path) {
    std::ifstream fin(in_path);
    if (!fin.is_open()) {
        std::cerr << "[ERROR] --eval: could not open " << in_path << std::endl;
        return false;
    }
    std::ofstream fout(out_path);
    if (!fout.is_open()) {
        std::cerr << "[ERROR] --eval: could not open " << out_path << " for writing" << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::istringstream iss(line);

        if (mode == "rmsnorm") {
            int8_t x[8], y[8];
            for (int i = 0; i < 8; i++) { int v; iss >> v; x[i] = (int8_t)v; }
            rmsnorm_hw_model(x, y);
            for (int i = 0; i < 8; i++) fout << (int)y[i] << (i == 7 ? "" : " ");
            fout << "\n";
        } else if (mode == "gelu") {
            int8_t x[8], y[8];
            for (int i = 0; i < 8; i++) { int v; iss >> v; x[i] = (int8_t)v; }
            gelu_hw_model(x, y);
            for (int i = 0; i < 8; i++) fout << (int)y[i] << (i == 7 ? "" : " ");
            fout << "\n";
        } else if (mode == "quant") {
            int64_t acc[4]; int32_t mult; int shift; int8_t y[4];
            for (int i = 0; i < 4; i++) iss >> acc[i];
            iss >> mult >> shift;
            for (int i = 0; i < 4; i++) y[i] = quant_model(acc[i], mult, shift);
            for (int i = 0; i < 4; i++) fout << (int)y[i] << (i == 3 ? "" : " ");
            fout << "\n";
        } else if (mode == "matmul") {
            int8_t A[4][8]; int8_t b[8]; int32_t c[4]; int32_t mult; int shift; int8_t y[4];
            for (int m = 0; m < 4; m++)
                for (int k = 0; k < 8; k++) { int v; iss >> v; A[m][k] = (int8_t)v; }
            for (int k = 0; k < 8; k++) { int v; iss >> v; b[k] = (int8_t)v; }
            for (int m = 0; m < 4; m++) iss >> c[m];
            iss >> mult >> shift;
            matmul_model(A, b, c, mult, shift, y);
            for (int m = 0; m < 4; m++) fout << (int)y[m] << (m == 3 ? "" : " ");
            fout << "\n";
        } else {
            std::cerr << "[ERROR] --eval: unsupported mode '" << mode << "'" << std::endl;
            return false;
        }
    }
    std::cout << "[C++ MODEL] --eval " << mode << ": wrote " << out_path << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "[ERROR] Must provide a block mode (e.g., './cpu_model exp' or './cpu_model ln')" << std::endl;
        return -1;
    }

    std::string mode = argv[1];

    // --eval <input_file> [<output_file>]: evaluate this mode's model function
    // over externally-supplied values instead of generating self-test vectors.
    if (argc >= 3 && std::string(argv[2]) == "--eval") {
        if (argc < 4) {
            std::cerr << "[ERROR] --eval requires an input file path" << std::endl;
            return -1;
        }
        std::string out_path = (argc >= 5) ? argv[4] : "eval_out.txt";
        return run_eval_mode(mode, argv[3], out_path) ? 0 : -1;
    }

    std::ofstream vec_file("inputs.txt");
    std::ofstream exp_file("expected.txt");

    if (!vec_file.is_open() || !exp_file.is_open()) {
        std::cerr << "[ERROR] Could not open IO files!" << std::endl;
        return -1;
    }

    if (mode == "rmsnorm") {
        const int N_VECS = 256;
        vec_file << N_VECS << "\n";
        srand(1337);
        for (int i = 0; i < N_VECS; i++) {
            int8_t x[8], y[8];
            for (int j = 0; j < 8; j++) {
                x[j] = (int8_t)((rand() % 256) - 128);
                vec_file << (int)x[j] << (j == 7 ? "" : " ");
            }
            vec_file << "\n";
            rmsnorm_hw_model(x, y);
            for (int j = 0; j < 8; j++) exp_file << (int)y[j] << (j == 7 ? "" : " ");
            exp_file << "\n";
        }
        std::cout << "[C++ MODEL] Generated RMSNORM test vectors." << std::endl;
    }
    else if (mode == "quant") {
        // Realistic (mult, shift) pairs map SOME expected max accumulator range
        // to ~127 (as calibration does); acc is generated relative to that
        // per-vector boundary so each vector gets a natural mix of in-range,
        // rounding-boundary, and saturating lanes -- not everything clamped.
        const int N_VECS = 256;
        vec_file << N_VECS << "\n";
        srand(1337);
        for (int i = 0; i < N_VECS; i++) {
            int shift = 4 + (rand() % 14);            // 4..17
            int32_t mult = 50 + (rand() % 200);       // 50..249
            double boundary = (double)((int64_t)1 << shift) * 127.0 / (double)mult;
            int64_t acc[4]; int8_t y[4];
            for (int j = 0; j < 4; j++) {
                double frac = ((rand() % 2600) - 500) / 1000.0;   // -0.5 .. 2.1
                acc[j] = (int64_t)(frac * boundary);
                vec_file << acc[j] << " ";
            }
            vec_file << mult << " " << shift << "\n";
            for (int j = 0; j < 4; j++) y[j] = quant_model(acc[j], mult, shift);
            for (int j = 0; j < 4; j++) exp_file << (int)y[j] << (j == 3 ? "" : " ");
            exp_file << "\n";
        }
        std::cout << "[C++ MODEL] Generated QUANT test vectors." << std::endl;
    }
    else if (mode == "matmul") {
        // A, b fully random int8 (natural accumulate magnitude is well below
        // the worst case 8*127*127, so a realistic (mult,shift) already gives a
        // natural mix of in-range/rounding/saturating results without needing
        // per-vector scaling like the quant case above).
        const int N_VECS = 256;
        vec_file << N_VECS << "\n";
        srand(1337);
        for (int i = 0; i < N_VECS; i++) {
            int8_t A[4][8]; int8_t b[8]; int32_t c[4];
            int shift = 14 + (rand() % 5);            // 14..18
            int32_t mult = 20 + (rand() % 130);       // 20..149
            int8_t y[4];
            for (int m = 0; m < 4; m++)
                for (int k = 0; k < 8; k++) {
                    A[m][k] = (int8_t)((rand() % 256) - 128);
                    vec_file << (int)A[m][k] << " ";
                }
            for (int k = 0; k < 8; k++) {
                b[k] = (int8_t)((rand() % 256) - 128);
                vec_file << (int)b[k] << " ";
            }
            for (int m = 0; m < 4; m++) {
                c[m] = (int32_t)((rand() % 2001) - 1000);
                vec_file << c[m] << " ";
            }
            vec_file << mult << " " << shift << "\n";
            matmul_model(A, b, c, mult, shift, y);
            for (int m = 0; m < 4; m++) exp_file << (int)y[m] << (m == 3 ? "" : " ");
            exp_file << "\n";
        }
        std::cout << "[C++ MODEL] Generated MATMUL test vectors." << std::endl;
    }
    else if (mode == "exp") {
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
            int8_t x[N_LANES], y[N_LANES];
            for (int j = 0; j < N_LANES; j++) {
                x[j] = (int8_t)((rand() % 256) - 128);
                vec_file << (int)x[j] << (j == N_LANES - 1 ? "" : " ");
            }
            vec_file << "\n";

            gelu_hw_model(x, y);
            for (int j = 0; j < N_LANES; j++)
                exp_file << (int)y[j] << (j == N_LANES - 1 ? "" : " ");
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