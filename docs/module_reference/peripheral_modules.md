# Peripheral Modules — `tr_nonlinear_vpu`

These modules live in [rtl_soc/tr_nonlinear_vpu/peripheral_modules/](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/).

---

## `piped_max`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/piped_max.sv](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/piped_max.sv)

### Operation

Finds the maximum signed value across `NUM_INPUTS` elements using a pipelined binary comparison tree. Inputs are fed into stage 0 of the tree; at each subsequent stage, pairs of values are compared and the larger is forwarded, reducing the element count by half per stage. The total latency is `STAGES = log2(NUM_INPUTS)` clock cycles. A valid shift-register tracks the pipeline so `valid_out` is asserted exactly `STAGES` cycles after `valid_in`.

> **Constraint:** `NUM_INPUTS` must be a power of two.

### Parameters

| Parameter    | Default | Description                          |
|-------------|---------|--------------------------------------|
| `NUM_INPUTS` | `8`     | Number of signed input elements      |
| `DATA_WIDTH` | `8`     | Bit width of each element            |

### Ports

| Port        | Direction | Width                      | Description                                  |
|-------------|-----------|----------------------------|----------------------------------------------|
| `clk`       | input     | 1                          | Clock                                        |
| `rst_n`     | input     | 1                          | Active-low synchronous reset                 |
| `valid_in`  | input     | 1                          | Asserted when `in_data` is valid             |
| `in_data`   | input     | `[DATA_WIDTH-1:0][NUM_INPUTS]` | Array of signed input values             |
| `max_out`   | output    | `DATA_WIDTH`               | Maximum signed value across all inputs       |
| `valid_out` | output    | 1                          | Asserted `STAGES` cycles after `valid_in`    |

---

## `scalar_sub`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/scalar_sub.sv](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/scalar_sub.sv)

### Operation

Combinationally subtracts a single scalar value (`sub_val`) from every element of an input vector. Each difference is computed using a sign-extended `DATA_WIDTH+1` bit intermediate to detect overflow, then clamped (saturated) to the representable signed range `[-(2^(DATA_WIDTH-1)), 2^(DATA_WIDTH-1)-1]`. For 8-bit data this is `[-128, +127]`. This module is purely combinational — no clock or reset.

### Parameters

| Parameter    | Default | Description                          |
|-------------|---------|--------------------------------------|
| `NUM_INPUTS` | `8`     | Number of elements in the vector     |
| `DATA_WIDTH` | `8`     | Bit width of each element            |

### Ports

| Port       | Direction | Width                           | Description                                        |
|------------|-----------|---------------------------------|----------------------------------------------------|
| `in_data`  | input     | `[DATA_WIDTH-1:0][NUM_INPUTS]`  | Signed input vector                                |
| `sub_val`  | input     | `DATA_WIDTH`                    | Signed scalar to subtract from every element       |
| `out_data` | output    | `[DATA_WIDTH-1:0][NUM_INPUTS]`  | Saturated result vector (`in_data[i] - sub_val`)   |

---

## `alpha_stabilizer`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/alpha_stabilizer.sv](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/alpha_stabilizer.sv)

### Operation

Piecewise-linear GELU sigmoid-slope scaler, forced into the negative domain `tr_exp_alu`'s Taylor-Region backbone supports. Scales `|x|` by one of four region-dependent coefficients (27, 26, 25, 24 sixteenths, re-scaled to `FRAC_W`), selected by which of the `|x|<1/2/3` real-valued bands `x` falls into — a piecewise refinement of the constant GELU sigmoid-slope factor (~1.702) for better accuracy across magnitude ranges. The result is saturated to the signed `W`-bit range, then unconditionally negated to non-positive (`out_vec = -|x_scaled|`) so it lands in the negative-only domain `shared_lut_rom`'s anchor table and `tr_exp_alu`'s Taylor polynomial expect, regardless of the original sign of `x`.

### Parameters

| Parameter | Default | Description                                                                 |
|-----------|---------|-------------------------------------------------------------------------------|
| `N`       | `8`     | Vector dimension                                                               |
| `W`       | `8`     | Word width of `in_vec`/`out_vec`                                               |
| `FRAC_W`  | `4`     | Fractional bits (defines the `|x|=1/2/3` region boundaries and coefficient scaling) |

### Ports

| Port      | Direction | Width           | Description                                  |
|-----------|-----------|------------------|-----------------------------------------------|
| `in_vec`  | input     | `[W-1:0][N]`      | Signed input vector (`x`)                     |
| `out_vec` | output    | `[W-1:0][N]`      | Non-positive, saturated, scaled `-|x|` result |

---

## `shared_lut_rom`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/shared_lut_rom.sv](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/shared_lut_rom.sv)

### Operation

8-entry `e^-k` anchor table shared by every `tr_exp_alu` lane. Fixed Q0.8, format-independent ROM holding `e^-1..e^-8` (the decay-only anchor points `tr_exp_alu`'s `round.sv` rounds `x` to). Each of the `N` lanes independently indexes the same 8-entry table via its own `a_idx` (only `a_idx`'s low 3 bits are used — the table itself does not grow with `LUT_IDX_W`).

### Parameters

| Parameter   | Default | Description                                                            |
|-------------|---------|--------------------------------------------------------------------------|
| `N`         | `8`     | Number of parallel lanes (independent index/lookup pairs)                 |
| `LUT_IDX_W` | `3`     | Width of each `a_idx` port; only bits `[2:0]` select the (always 8-entry) table |

### Ports

| Port    | Direction | Width               | Description                                    |
|---------|-----------|---------------------|--------------------------------------------------|
| `a_idx` | input     | `[LUT_IDX_W-1:0][N]` | Indices requested by the ALUs                     |
| `e_a`   | output    | `[7:0][N]`           | Anchors returned to the ALUs (Q0.8, format-independent) |

---

## `symmetry_modifier`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/symmetry_modifier.sv](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/symmetry_modifier.sv)

### Operation

GELU sigmoid symmetry logic (vectorized) — a combinational trick that executes `Out = (x_raw < 0) ? (1.0 - y_sig) : y_sig`. Used to restore the full GELU shape from the non-positive TR-EXP domain: since the shared backbone can only evaluate `exp` for non-positive arguments, GELU's sigmoid is evaluated on `|x|` and this module restores the correct value for negative `x` via `1 - sigmoid(|x|) = sigmoid(-|x|)`. Extracts the sign bit directly from the raw input vector.

### Parameters

| Parameter | Default | Description                                          |
|-----------|---------|---------------------------------------------------------|
| `N`       | `8`     | Vector dimension                                           |
| `WIDTH_X` | `8`     | Word width of the raw input `x` (to check the sign bit)    |
| `WIDTH_Y` | `8`     | Word width of the sigmoid input/output                     |
| `FRAC_W`  | `4`     | Fractional bit width of `y` (defines the `1.0` constant)   |

### Ports

| Port            | Direction | Width               | Description                                  |
|-----------------|-----------|---------------------|-------------------------------------------------|
| `x_raw`         | input     | `[WIDTH_X-1:0][N]`   | Original (pre-`|x|`) signed input, for its sign  |
| `y_sig`         | input     | `[WIDTH_Y-1:0][N]`   | Sigmoid value computed on `|x|`                  |
| `mode_en`       | input     | 1                    | 1: apply symmetry correction, 0: bypass          |
| `sig_corrected` | output    | `[WIDTH_Y-1:0][N]`   | Symmetry-corrected sigmoid result                |
