#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Single source of truth for all the bit-exact hardware math models
// (exp/ln primitives, rmsnorm_hw_model,
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
            // line: "log2n x0 .. x(N-1)" -> "inv_rms y0 .. y(N-1)"  (tiled flow, see rms_* in tr_math_model.hpp)
            int log2n; iss >> log2n;
            int n = 1 << log2n;
            std::vector<int8_t> x(n), y(n);
            for (int i = 0; i < n; i++) { int v; iss >> v; x[i] = (int8_t)v; }
            int32_t acc = 0;
            for (int t = 0; t < n / 8; t++) acc = rms_acc_tile(&x[8 * t], acc, t == 0);
            uint32_t inv = rms_scale(acc, log2n);
            for (int t = 0; t < n / 8; t++) rms_apply_tile(&x[8 * t], inv, &y[8 * t]);
            fout << inv;
            for (int i = 0; i < n; i++) fout << " " << (int)y[i];
            fout << "\n";
        } else if (mode == "swiglu") {
            // line: "n g0..g(n-1) u0..u(n-1)" -> "e y0..y(n-1)"  (two-pass SwiGLU row, see sw_* in tr_math_model.hpp)
            int n; iss >> n;
            std::vector<int8_t> g(n), u(n), y(n);
            for (int i = 0; i < n; i++) { int v; iss >> v; g[i] = (int8_t)v; }
            for (int i = 0; i < n; i++) { int v; iss >> v; u[i] = (int8_t)v; }
            int32_t mx = 0;
            for (int i = 0; i < n; i++) { int32_t a = sw_abs_prod(sw_silu(g[i]), u[i]); if (a > mx) mx = a; }
            int e = sw_gain(mx);
            for (int i = 0; i < n; i++) y[i] = sw_apply(sw_silu(g[i]), u[i], e);
            fout << e;
            for (int i = 0; i < n; i++) fout << " " << (int)y[i];
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
        // PRODUCTION tiled RMSNorm: one row per line. Input line "log2n x0 .. x(N-1)" (N = 1<<log2n),
        // expected line "inv_rms y0 .. y(N-1)". The TB runs cmd 7/8 per tile (sum of squares), cmd 9
        // once, cmd 0xA per tile; inv_rms is read from the controller register. Row mix: random
        // full-range, heavy-tailed, outlier, tiny (growth branch), constant, and sparse rows, at
        // N = 8 .. 2048. Most rows have max|x| = 127 (what the per-token quantizer produces).
        const int N_ROWS = 160;
        vec_file << N_ROWS << "\n";
        srand(1337);
        for (int r = 0; r < N_ROWS; r++) {
            static const int L2[] = {3, 3, 4, 5, 6, 7, 8, 9, 10, 11};
            int log2n = L2[r % 10];
            int n = 1 << log2n;
            int kind = (r / 10) % 8;
            std::vector<int8_t> x(n), y(n);
            for (int i = 0; i < n; i++) {
                int v;
                switch (kind) {
                    case 0: v = (rand() % 256) - 128; break;                                   // uniform
                    case 1: v = ((rand() % 41) - 20) + ((rand() % 41) - 20) + ((rand() % 41) - 20); break;  // bell
                    case 2: v = (rand() % 7) - 3; break;                                       // tiny -> growth
                    case 3: v = (rand() % 3) - 1; break;                                       // tinier
                    case 4: v = 40; break;                                                     // constant
                    case 5: v = (rand() % 50 == 0) ? ((rand() % 255) - 127) : ((rand() % 5) - 2); break;  // sparse
                    case 6: v = (rand() % 2) ? 127 : -128; break;                              // saturated
                    default: v = (rand() % 61) - 30; break;
                }
                x[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
            }
            if (kind == 1 || kind == 5 || kind == 7) x[rand() % n] = 127;                      // quantizer: max|x| = 127
            vec_file << log2n;
            for (int i = 0; i < n; i++) vec_file << " " << (int)x[i];
            vec_file << "\n";
            int32_t acc = 0;
            for (int t = 0; t < n / 8; t++) acc = rms_acc_tile(&x[8 * t], acc, t == 0);
            uint32_t inv = rms_scale(acc, log2n);
            for (int t = 0; t < n / 8; t++) rms_apply_tile(&x[8 * t], inv, &y[8 * t]);
            exp_file << inv;
            for (int i = 0; i < n; i++) exp_file << " " << (int)y[i];
            exp_file << "\n";
        }
        std::cout << "[C++ MODEL] Generated tiled RMSNORM test vectors." << std::endl;
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
        // tr_ln_alu (PWL datapath), exhaustive over the WIDTH-bit input domain. Config via env
        // LN_CFG="WIDTH,BITS,SEGMENTS,COEF_BITS,OUT_WIDTH" (default 16,4,2,8,8 = the TB defaults).
        int W = 16, B = 4, S = 2, C = 8, O = 8;
        if (const char* e = getenv("LN_CFG")) sscanf(e, "%d,%d,%d,%d,%d", &W, &B, &S, &C, &O);
        vec_file << (1 << W) << "\n";
        for (int xq = 0; xq < (1 << W); xq++) {
            vec_file << xq << "\n";
            exp_file << xq << " " << tr_pwl_ln_scalar(xq, B, S, C, O) << "\n";
        }
        std::cout << "[C++ MODEL] Generated LN test vectors (W=" << W << " B=" << B << " SEG=" << S
                  << " C=" << C << " OUT=" << O << ")." << std::endl;
    } 
    else if (mode == "softmax") {
        // PRODUCTION path: tr_soc_top_int CMD=0x01 (SM_P1..SM_P4), bit-exact via softmax_hw_model_ex. Output is
        // Q0.8 UNSIGNED (0..255, saturating); expected line = 8 probabilities + the tile's z = sat8(max + ln(S))
        // (read back via ADDR_SM_Z), which the CPU needs to combine tiles (logsumexp over z).
        // Tile mix: full-range random (as before), attention-like tiles (scores around a head's range, some lanes
        // masked at -128), fully masked tiles, equal-max tiles, near-saturating max (z clamps at 127).
        const int N_VECS = 384;
        vec_file << N_VECS << "\n";
        srand(1337);
        for (int i = 0; i < N_VECS; i++) {
            int8_t x[8]; uint8_t y[8]; int8_t z;
            int kind = i < 256 ? 0 : 1 + (i - 256) % 5;
            for (int j = 0; j < 8; j++) {
                int v;
                switch (kind) {
                    case 0: v = (rand() % 256) - 128; break;                                       // uniform
                    case 1: v = (rand() % 5 == 0) ? -128 : (rand() % 121) - 60; break;             // attention-like, some masked
                    case 2: v = -128; break;                                                       // fully masked tile
                    case 3: v = 20; break;                                                         // all equal
                    case 4: v = 90 + (rand() % 38); break;                                         // high scores: z clamps
                    default: v = (j < 1 + (i % 7)) ? (rand() % 80) - 40 : -128; break;             // causal-prefix tile
                }
                x[j] = (int8_t)v;
                vec_file << (int)x[j] << (j == 7 ? "" : " ");
            }
            vec_file << "\n";

            softmax_hw_model_ex(x, y, &z);
            for (int j = 0; j < 8; j++) exp_file << (int)y[j] << " ";
            exp_file << (int)z << "\n";
        }
        std::cout << "[C++ MODEL] Generated SOFTMAX (production tr_soc_top_int path, probs + z) vectors." << std::endl;
    }
    else if (mode == "swiglu") {
        // PRODUCTION two-pass SwiGLU (cmds 5/6/0xB/0xC, host-sequenced in 8-lane groups). One row per line:
        // input "n g0..g(n-1) u0..u(n-1)" (gate / up int8 codes), expected "e y0..y(n-1)" with e the row gain
        // read back from the controller. Row mix: heavy-tailed gate/up (what the FFN produces), gate outliers
        // at the int8 limits, an all-zero up row (e = 10, the gain cap), a tiny-product row, saturating rows;
        // n = 8 .. 8192 (Llama-3.2-1B intermediate size).
        static const int NS[] = {8, 8, 16, 64, 256, 256, 1024, 1024, 8192, 8192};
        const int N_ROWS = 40;
        vec_file << N_ROWS << "\n";
        srand(2024);
        auto tdist = [](double scale) {                       // heavy-tailed int8 sample
            double a = (rand() / (double)RAND_MAX - 0.5) * 2, b = (rand() / (double)RAND_MAX - 0.5) * 2;
            double v = (a * a * a + 0.5 * b) * scale;
            int r = (int)(v < 0 ? v - 0.5 : v + 0.5);
            return r > 127 ? 127 : (r < -128 ? -128 : r);
        };
        for (int r = 0; r < N_ROWS; r++) {
            int n = NS[r % 10];
            int kind = (r / 10) % 4 + (r % 7 == 0 ? 4 : 0);
            std::vector<int8_t> g(n), u(n), y(n);
            for (int i = 0; i < n; i++) {
                int gv, uv;
                switch (kind) {
                    case 0: gv = tdist(40);  uv = tdist(60);  break;
                    case 1: gv = tdist(100); uv = tdist(100); break;
                    case 2: gv = (rand() % 256) - 128; uv = (rand() % 256) - 128; break;   // uniform
                    case 3: gv = tdist(8);   uv = tdist(6);   break;                        // small
                    case 4: gv = -128 + (rand() % 8); uv = (rand() % 5) - 2; break;          // tiny products (gate ~ -8)
                    case 5: gv = (rand() % 8 == 0) ? 127 : tdist(30); uv = 127; break;       // saturating
                    case 6: gv = tdist(60);  uv = 0;          break;                        // all-zero up -> e = 10 (cap)
                    default: gv = tdist(20); uv = (rand() % 3) - 1; break;
                }
                g[i] = (int8_t)gv; u[i] = (int8_t)uv;
            }
            vec_file << n;
            for (int i = 0; i < n; i++) vec_file << " " << (int)g[i];
            for (int i = 0; i < n; i++) vec_file << " " << (int)u[i];
            vec_file << "\n";
            int32_t mx = 0;
            for (int i = 0; i < n; i++) { int32_t a = sw_abs_prod(sw_silu(g[i]), u[i]); if (a > mx) mx = a; }
            int e = sw_gain(mx);
            for (int i = 0; i < n; i++) y[i] = sw_apply(sw_silu(g[i]), u[i], e);
            exp_file << e;
            for (int i = 0; i < n; i++) exp_file << " " << (int)y[i];
            exp_file << "\n";
        }
        std::cout << "[C++ MODEL] Generated two-pass SWIGLU test vectors." << std::endl;
    }
    else {
        std::cerr << "[ERROR] Unknown mode: " << mode << std::endl;
        return -1;
    }

    vec_file.close();
    exp_file.close();
    return 0;
}