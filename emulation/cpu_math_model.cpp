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
    else {
        std::cerr << "[ERROR] Unknown mode: " << mode << std::endl;
        return -1;
    }

    vec_file.close();
    exp_file.close();
    return 0;
}