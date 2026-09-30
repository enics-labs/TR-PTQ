# TR Backbone — `tr_nonlinear_vpu`

These modules live in [rtl_soc/tr_nonlinear_vpu/tr_backbone/](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/). Together they form the shared log-domain math pipeline (`tr_backbone_wrapper`) that every GELU/Softmax/RMSNorm pass routes through: pre-conditioner → `tr_ln_alu` → post-conditioner → `tr_exp_alu` (+ its `round`/`quadratic_divider` helpers).

---

## `tr_backbone_wrapper`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_backbone_wrapper.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_backbone_wrapper.sv)

### Operation

Chains `pre_ln_modifier` → `tr_ln_alu` → `post_ln_modifier` → `tr_exp_alu` into one purely combinational pipeline. Expects wide inputs and natively steps precision down through `tr_ln_alu`. This is the module `tr_nonlinear_vpu`'s crossbar routes every GL_P*/SM_P*/RM_P* pass's log-domain math through.

### Parameters

| Parameter   | Default | Description                                             |
|-------------|---------|-----------------------------------------------------------|
| `N`         | `8`     | Vector dimension                                         |
| `WIDTH_IN`  | `16`    | Incoming datapath width                                  |
| `WIDTH_OUT` | `8`     | Target operating width for exp/mantissa                  |
| `FRAC_W`    | `4`     | Fractional bit width                                     |
| `LUT_IDX_W` | `3`     | Width of the ROM index for the Taylor anchor              |

### Ports

| Port           | Direction | Width                          | Description                                    |
|----------------|-----------|---------------------------------|-------------------------------------------------|
| `clk`, `rst_n` | input     | 1                                | Clock / active-low reset                        |
| `in_valid`     | input     | 1                                | Input valid                                     |
| `out_valid`    | output    | 1                                | Output valid                                    |
| `bypass_ln`    | input     | 1                                | Skip `tr_ln_alu` (route straight to exp)        |
| `mode_pre_ln`  | input     | 1                                | `pre_ln_modifier`'s +1.0 select                 |
| `mode_post_ln` | input     | 2                                | `post_ln_modifier`'s bypass/÷/InvSqrt select    |
| `vec_in`       | input     | `[WIDTH_IN-1:0][N]`              | Input vector                                    |
| `log_out`      | output    | `[WIDTH_OUT-1:0][N]`             | `tr_ln_alu`'s (post-conditioned) log output     |
| `a_idx_out`    | output    | `[LUT_IDX_W-1:0][N]`             | Shared-LUT anchor index for `shared_lut_rom`    |
| `mantisa_out`  | output    | `[WIDTH_OUT-1:0][N]`             | Taylor-mantissa term to multiply against `e_a`  |
| `is_zero_out`  | output    | `[N]`                            | Whether this lane's `exp` result should be 0    |

---

## `tr_ln_alu`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_ln_alu.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_ln_alu.sv)

### Operation

Integer natural-logarithm approximator via MSB-exponent + linear-mantissa log2 decomposition. Finds `xq`'s MSB (`floor(log2(xq))`) using a balanced binary-search tree (not a linear scan — this was a real critical-path fix, see the FPGA M2 timing results), normalizes `xq` so that bit sits at the top of the word, and takes its top bits as a linear mantissa term (`log2(1.f) ≈ f` over `[0,1)`). Combines the exponent and mantissa terms into `log2(xq)`, then converts to `ln` by multiplying by the fixed-point constant `≈ln(2)` (via shifts, no real multiplier), saturating to `OUT_WIDTH`. `xq==0` short-circuits to `yq=0`.

### Parameters

| Parameter   | Default | Description                                    |
|-------------|---------|--------------------------------------------------|
| `WIDTH`     | `16`    | Input width (`xq`)                              |
| `BITS`      | `4`     | Fractional bits of `xq`                          |
| `OUT_WIDTH` | `8`     | Output width (`yq`), saturated                   |

### Ports

| Port  | Direction | Width       | Description                       |
|-------|-----------|-------------|-------------------------------------|
| `xq`  | input     | `WIDTH`     | Unsigned input value                |
| `yq`  | output    | `OUT_WIDTH` | Signed `ln(xq)`, saturated           |

---

## `tr_exp_alu`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_exp_alu.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_exp_alu.sv)

### Operation

Taylor-Region exponential ALU: approximates `exp(x)` (`x` expected `<= 0`) via nearest-LUT-anchor lookup plus a local Taylor polynomial in the residual `delta = x - a`. `round.sv` rounds `x` to the nearest of `shared_lut_rom`'s anchor points, returning the LUT index (for the caller to fetch `e^a`), `is_zero` (underflow anchor), and the sign of `delta`. `quadratic_divider` computes the `delta²/2` term. The Taylor mantissa — truncated to order `ITER` (0: constant, 1: linear, 2: quadratic) — is returned for the caller to multiply against `shared_lut_rom`'s `e_a`, reconstructing `exp(x) ≈ e_a · mantissa`.

### Parameters

| Parameter   | Default | Description                                                |
|-------------|---------|---------------------------------------------------------------|
| `WIDTH`     | `8`     | I/O word width (`x`, `mantisa`)                               |
| `FRAC_W`    | `4`     | Fractional bits of `x` (also the Taylor delta's width)        |
| `LUT_IDX_W` | `3`     | Width of the shared LUT index (`a_idx`)                       |
| `ITER`      | `2`     | Taylor truncation order: 0=constant, 1=linear, 2=quadratic    |

### Ports

| Port      | Direction | Width         | Description                               |
|-----------|-----------|----------------|--------------------------------------------|
| `x`       | input     | `WIDTH`        | Signed input, expected `<= 0`              |
| `a_idx`   | output    | `LUT_IDX_W`     | Anchor index, sent to `shared_lut_rom`     |
| `mantisa` | output    | `WIDTH`         | Taylor mantissa term                        |
| `is_zero` | output    | 1               | Whether `x` underflowed to the 0 anchor    |

---

## `round`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/round.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/round.sv)

### Operation

Rounds a fixed-point input to its nearest integer anchor and generates the corresponding `shared_lut_rom` index. Returns `is_ceil` (whether rounding moved `x` up past its anchor — the sign of the residual `delta = x - anchor` `tr_exp_alu`'s Taylor polynomial needs) and `is_zero` (rounded integer is exactly 0). `lut_idx` is generated in one of two modes (`DECAY_ONLY_LUT`): mode 1 (default) maps the negative-only decay anchors onto a 0-based index for `shared_lut_rom`'s 8-entry decay table, saturating at the highest index once the magnitude exceeds what `LUT_IDX_W` bits can distinguish; mode 2 passes the rounded integer straight through as a signed index, for the bidirectional (positive and negative anchors) LayerNorm-style case.

### Parameters

| Parameter        | Default | Description                                                      |
|------------------|---------|----------------------------------------------------------------------|
| `WIDTH`          | `8`     | Input width (`x`)                                                    |
| `FRAC_W`         | `4`     | Fractional bits of `x`                                               |
| `LUT_IDX_W`      | `3`     | Width of the generated LUT index                                     |
| `DECAY_ONLY_LUT` | `1`     | 1: decay-only anchor table (`tr_exp_alu` default); 0: bidirectional  |

### Ports

| Port      | Direction | Width       | Description                              |
|-----------|-----------|-------------|---------------------------------------------|
| `x`       | input     | `WIDTH`     | Signed fixed-point input                    |
| `is_zero` | output    | 1           | Rounded integer is exactly 0                |
| `is_ceil` | output    | 1           | Rounding moved `x` up past its anchor       |
| `lut_idx` | output    | `LUT_IDX_W` | Index for `shared_lut_rom`                  |

---

## `quadratic_divider`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/quadratic_divider.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/quadratic_divider.sv)

### Operation

Computes the `delta²/2` quadratic term of `tr_exp_alu`'s Taylor polynomial for `exp(x) ≈ e_a·(1+delta+delta²/2)`. At `FRAC_W==4` this uses exact K-map equations over `delta[3:0]` instead of a real multiplier (mode 1); any other `FRAC_W` falls back to a generic signed multiply (mode 2, e.g. for LayerNorm's wider fractional format). Both modes are mathematically identical at `FRAC_W==4` regardless of `WIDTH` — the gate is `FRAC_W==4` alone, not tied to any particular `WIDTH`.

### Parameters

| Parameter | Default | Description                                                        |
|-----------|---------|------------------------------------------------------------------------|
| `WIDTH`   | `8`     | Nominal word width this term feeds into (doesn't size any signal here) |
| `FRAC_W`  | `4`     | Fractional bits of `delta` (also selects mode 1 vs mode 2)             |

### Ports

| Port       | Direction | Width    | Description                       |
|------------|-----------|----------|--------------------------------------|
| `delta`    | input     | `FRAC_W` | Signed residual (`x - anchor`)       |
| `quad_out` | output    | `FRAC_W` | `(delta²) >> (FRAC_W+1)`             |

---

## `pre_ln_modifier`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/pre_ln_modifier.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/pre_ln_modifier.sv)

### Operation

Log-domain pre-conditioner (vectorized), primarily used for GELU denominator construction. Computes `out = mode_add_one ? x + 1.0 : x`, implemented as a zero-cost bit operation at the integer boundary (no real adder needed for the `+1.0` case beyond the fixed constant add).

### Parameters

| Parameter  | Default | Description                                              |
|------------|---------|--------------------------------------------------------------|
| `N`        | `8`     | Vector dimension                                              |
| `WIDTH_IN` | `8`     | Word width of `x_in`/`y_out`                                  |
| `FRAC_W`   | `4`     | Fractional bits (defines the location of the `1.0` bit)       |

### Ports

| Port           | Direction | Width                   | Description                          |
|----------------|-----------|--------------------------|-----------------------------------------|
| `x_in`         | input     | `[WIDTH_IN-1:0][N]`      | Signed input vector                     |
| `mode_add_one` | input     | 1                        | 0: bypass, 1: add `+1.0`                |
| `y_out`        | output    | `[WIDTH_IN-1:0][N]`      | Result vector                           |

---

## `post_ln_modifier`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/post_ln_modifier.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/post_ln_modifier.sv)

### Operation

Log-domain division and InvSqrt logic (vectorized) — the scalar multiplications the pipeline needs in the log domain to avoid physical dividers. Mode 00: bypass (`x`). Mode 01: division (`-1.0 * x`, standard two's-complement negation — this is how Softmax/GELU reciprocals are computed without a divider). Mode 10: inverse square root (`-0.5 * x`, arithmetic right shift then negate — this is RMSNorm's `1/sqrt` step).

### Parameters

| Parameter | Default | Description                                    |
|-----------|---------|-----------------------------------------------------|
| `N`       | `8`     | Vector dimension                                     |
| `W`       | `20`    | Word width (often wider, since it follows `tr_ln_alu`) |

### Ports

| Port       | Direction | Width           | Description                                      |
|------------|-----------|------------------|-----------------------------------------------------|
| `x_in`     | input     | `[W-1:0][N]`     | Signed input vector                                 |
| `mode_sel` | input     | 2                | 00: bypass, 01: `-1.0x` (÷), 10: `-0.5x` (InvSqrt)  |
| `y_out`    | output    | `[W-1:0][N]`     | Result vector                                       |
