#include <iostream>
#include <fstream>
#include <vector>

// 1. Redefine CUDA keywords and hardware intrinsics to compile locally on CPU
#ifndef __CUDACC__
    #define __device__ 
    #define __forceinline__ inline
    #define __constant__ const
    
    // Map CUDA's count-leading-zeros to GCC's native CPU equivalent
    #define __clz(x) __builtin_clz(x)
#endif

// 2. Define the LUT constant
const uint8_t EXP_LUT_CONST[9] = {0, 0, 1, 2, 5, 13, 35, 94, 255};

// 3. Updated include path based on your submodule location
#include "../emulation/stable_code/mrcp_quant/optimized_layers/common/tr_math.cuh"

int main() {
    std::ofstream vec_file("input_vectors.txt");
    std::ofstream exp_file("expected_exp_out.txt");

    if (!vec_file.is_open() || !exp_file.is_open()) {
        std::cerr << "[ERROR] Could not open files for writing!" << std::endl;
        return -1;
    }

    // Update the header to reflect the 129 valid architectural inputs
    vec_file << "129\n";

    // ONLY sweep the valid architectural domain (-128 to 0)
    for (int i = -128; i <= 0; i++) {
        int z_q44 = i;
        
        int expected_y = tr_approx_exp_scalar_2rd(z_q44);

        vec_file << z_q44 << "\n";
        exp_file << z_q44 << " " << expected_y << "\n";
    }

    vec_file.close();
    exp_file.close();
    
    std::cout << "[C++ MODEL] Successfully generated 129 valid test vectors." << std::endl;
    return 0;
}