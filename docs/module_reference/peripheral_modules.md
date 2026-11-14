# Peripheral Modules — `tr_nonlinear_vpu`

These modules live in [rtl_soc/tr_nonlinear_vpu/peripheral_modules/](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/).

---

## `piped_max`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/piped_max.sv](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/piped_max.sv)

### Operation

Finds the maximum signed value across `NUM_INPUTS` elements using a pipelined binary comparison tree. Inputs are fed into stage 0 of the tree; at each subsequent stage, pairs of values are compared and the larger is forwarded, reducing the element count by half per stage. The total latency is `STAGES = log2(NUM_INPUTS)` clock cycles. A valid shift-register tracks the pipeline so `valid_out` is asserted exactly `STAGES` cycles after `valid_in`.

> **Constraint:** `NUM_INPUTS` must be a power of two.
>
> **Shared use:** `tr_nonlinear_vpu` instantiates it once at `DATA_WIDTH = WIDE_W` (20) behind an input mux (`mux_max_sel`): softmax's row max (sign-extended int8 lanes) and SwiGLU's max `|silu·up|` (non-negative wide products). The softmax result uses only the low 8 bits.

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

## `sigma_rom`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/sigma_rom.sv](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/sigma_rom.sv)

### Operation

SwiGLU sigmoid lookup: `sigma(x/16)` in Q.6 for every int8 gate code `x` (combinational, one lookup per lane). `silu(x) = x · sigma(x)` needs sigma only at the 256 int8 gate codes. `sigma(-x) = 1 - sigma(x)` and neighbouring codes share an entry, so the table has 64 entries of 7 bits (values reach 64 = 1.0) indexed by `|x| >> 1` (`|x| = 128` reads the last entry); negative codes read `64 - entry`. The entries are `sigma((2b + 1/2)/16)` rounded to 6 fractional bits. **Generated** by `tools/scripts/llama32_1b/gen_rom_sv.py` from `tools/models/vpu_ops.py::_sigma_table_sym`; the `swiglu` regression checks it against the C++ golden `sw2_silu`. The earlier 256 × 6 b table of the alpha → exp → ln chain gives the same perplexity within noise (`docs/llama/accuracy-study.md`).

### Parameters

| Parameter | Default | Description      |
|-----------|---------|------------------|
| `N`       | `8`     | Vector dimension |

### Ports

| Port    | Direction | Width        | Description                       |
|---------|-----------|--------------|-----------------------------------|
| `x`     | input     | `[7:0][N]`   | Signed int8 gate codes (Q4.4)     |
| `sigma` | output    | `[6:0][N]`   | Unsigned `sigma(x/16)` in Q.6 (0..64) |

---

## `rsq_unit`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/rsq_unit.sv](../../rtl_soc/tr_nonlinear_vpu/peripheral_modules/rsq_unit.sv)

### Operation

Reciprocal-sqrt unit of RMSNorm. The shared `tr_ln_alu` on lane 0 receives the rounded mean `mk = 2^k (1 + f)` and finds `k` (`msb`) and the normalized mantissa fraction `f` (`fk`, 6 bits) on its way to the ln; `tr_ln_alu` exports both (`msb_o`, `fk_o`) and `tr_backbone_wrapper` registers them next to `out_valid` (`ln_msb_out`, `ln_fk_out`). `rsq_unit` reads a 2 × 32-entry table of 12-bit entries (one integer bit) indexed by `{k & 1, fk[5:1]}`, `entry = round(2^11 · c / sqrt(1 + (i + 1/2)/32))` with `c = 1` (even `k`) or `sqrt 2` (odd `k`), and shifts it right by `(k + 1) >> 1` with one half-up rounding. The result `inv = round(2^11 / sqrt(mk))` is the wide `inv_rms` register (12 bits, 8 fractional bits); no cap is needed (the right shift keeps it below 2^12). It replaces the former exp of `-0.5 ln(mean)` with its sign-guarded growth branch and the `recip_rom` table. **Generated** by `tools/scripts/llama32_1b/gen_rom_sv.py` from `tools/models/vpu_ops.py::rsq_inv_mk`; bit-exact to `tr_math_model.hpp::rms2_inv_mk`, checked over every reachable `mk` (0..8191) by `verify_block.py rsq`.

### Parameters

| Parameter | Default | Description                                           |
|-----------|---------|-------------------------------------------------------|
| `MSB_W`   | `4`     | Width of `msb`                                        |
| `BITS`    | `6`     | Width of `fk` (the top 5 bits index the table)        |
| `INV_W`   | `12`    | Width of the inverse-rms register                     |

### Ports

| Port  | Direction | Width         | Description                          |
|-------|-----------|---------------|--------------------------------------|
| `msb` | input     | `[MSB_W-1:0]` | Leading-one position of `mk`         |
| `fk`  | input     | `[BITS-1:0]`  | Mantissa bits below the leading one  |
| `inv` | output    | `[INV_W-1:0]` | Inverse-rms register value           |

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
