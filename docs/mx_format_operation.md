# MX Format SoC Operation

## Overview
The Transformer Vector Processing Unit (VPU) utilizes an MX (Microscaling) format to perform deep learning acceleration. The MX format dynamically adapts the fractional precision of data by assigning a shared exponent block across a vector of mantissas, combining the hardware efficiency of low-precision integers (like INT8) with the dynamic range of floating-point arithmetic.

## Operation Flow

### 1. Matrix Multiplication (Linear Phase)
The execution flow begins with the linear layers via the `dot_product_engine` (`mac_array_engine`). Computations here perform standard integer accumulation into large 32-bit registers (`ACC_W = 32`) to prevent overflow. During this phase, exponents for the Activation and Weight arrays are simply added together (`base_exp = exp_act_in + exp_weight_in`).

### 2. The MX Requantizer
After the linear accumulation, the 32-bit wide accumulators must be compressed back down to 8-bit mantissas to fit in standard SRAM or the non-linear math engines. The `requantize_engine_mx` is responsible for this:
- **Maximum Absolute Value:** It scans the `N` parallel accumulators to find the maximum absolute value.
- **Leading Zero Count:** It calculates how many bits of headroom exist.
- **Dynamic Compression:** It shifts the 32-bit accumulators rightward until the largest value perfectly fits into the 8-bit mantissa space (7 magnitude bits + 1 sign bit).
- **Exponent Update:** The number of right-shifts is added to the base exponent, producing the final `mx_shared_exp`.

### 3. Non-Linear VPU Expansion
Before the compressed 8-bit mantissas enter the non-linear VPU (which evaluates functions like GELU, Softmax, and RMSNorm), they pass through the `dynamic_shifter_mx`. 
- **True-Value Restoration:** If a non-linear operation is scale-variant (like GELU or Softmax), the mantissas are shifted *left* by `mx_shared_exp`. This expands them into wider 16-bit registers (`VPU_W = 16`), restoring their "true" values within the Q12.4 fixed-point domain.
- **Scale-Invariant Bypass:** For operations like RMSNorm, which are inherently scale-invariant ($RMS(x) = x / RMS(x)$), this expansion is bypassed. The mantissas enter un-expanded to avoid arithmetic overflow when squared in the MAC engines.

### 4. Mathematical Execution (Fixed-Point)
Inside the VPU, computations are resolved using Q12.4 fixed-point math:
- **GELU:** Applies piece-wise combinations via the Symmetry Modifier and Backbone LUTs.
- **Softmax:** Utilizes a multi-pass pipeline. 
  1. Finds the maximum of the expanded vector.
  2. Subtracts the max ($x_i - max$).
  3. Computes the exponent ($e^{x_i - max}$).
  4. Sums the exponents and calculates the inverse-log.
  5. Multiplies to yield the final probability distribution.
- **RMSNorm:** Computes the variance, calculates the inverse root using the log backbone, and multiplies back against the input vector.

### 5. The MX Formatter (Final Compression)
Once the VPU finishes the calculation on the wider datapath, the results must be stored back into the 8-bit SRAM. The `formatter_mx`:
- Evaluates the VPU outputs.
- Finds the maximum magnitude to determine the necessary right-shift.
- Generates a new `E_out` exponent.
- Compresses the outputs down into final 8-bit mantissas for storage.

This pipelined approach ensures maximum mathematical precision inside the math engines while maintaining compact 8-bit memory footprints globally.