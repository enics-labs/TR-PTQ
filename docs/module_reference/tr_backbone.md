# TR Backbone — `tr_nonlinear_vpu`

These modules live in [rtl_soc/tr_nonlinear_vpu/tr_backbone/](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/). Together they form the shared log-domain math pipeline (`tr_backbone_wrapper`) that every Softmax/RMSNorm pass routes through: `tr_ln_alu` → post-conditioning (bypass or rounded −0.5×) → `tr_exp_alu` (+ its `round`/`quadratic_divider` helpers).

---

## `tr_backbone_wrapper`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_backbone_wrapper.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_backbone_wrapper.sv)

### Operation

Chains `tr_ln_alu` → post-ln mode → `tr_exp_alu` into a three-register-stage pipeline (input regs; `tr_ln_alu` → regs; round-back/saturate + post-ln mode → regs; `tr_exp_alu` combinational to the outputs — the extra register after the ln closes timing at 3.3 ns, 3 cycles `in_valid` → `out_valid`). Expects wide inputs and natively steps precision down through `tr_ln_alu`. **One shared ln unit for every operator:** `tr_ln_alu` runs at `BITS = FRAC_W + LN_EXTRA` (6) with an `(WIDTH_IN + LN_EXTRA)`-bit input. Operators with `FRAC_W` fractional bits feed `x << LN_EXTRA` (`ln_pre_scaled = 0`); RMSNorm feeds its mean already at `FRAC_W + LN_EXTRA` bits (`ln_pre_scaled = 1`). The Q.6 result is rounded back to Q.`FRAC_W` and saturated to `WIDTH_OUT` (software model `tr_vpu_ln` in `tr_math_model.hpp`). This is the module `tr_nonlinear_vpu`'s crossbar routes every SM_P*/RM_P* pass's log-domain math through.

### Parameters

| Parameter   | Default | Description                                             |
|-------------|---------|-----------------------------------------------------------|
| `N`         | `8`     | Vector dimension                                         |
| `WIDTH_IN`  | `16`    | Incoming datapath width                                  |
| `WIDTH_OUT` | `8`     | Target operating width for exp/mantissa                  |
| `FRAC_W`    | `4`     | Fractional bit width                                     |
| `LUT_IDX_W` | `3`     | Width of the ROM index for the Taylor anchor              |
| `LN_EXTRA`  | `2`     | Extra fractional bits of the shared ln unit (`BITS = FRAC_W + LN_EXTRA`) |
| `LN_IN_W`   | `13`    | Width of the shared ln unit's input. The largest reachable value is 4096 (RMSNorm mean with all codes −128); larger inputs are clamped (unreachable with the shipped controller) |

### Ports

| Port           | Direction | Width                          | Description                                    |
|----------------|-----------|---------------------------------|-------------------------------------------------|
| `clk`, `rst_n` | input     | 1                                | Clock / active-low reset                        |
| `in_valid`     | input     | 1                                | Input valid                                     |
| `out_valid`    | output    | 1                                | Output valid                                    |
| `bypass_ln`    | input     | 1                                | Skip `tr_ln_alu` (route straight to exp)        |
| `mode_post_ln` | input     | 2                                | `00`: bypass; `11`: rounded −0.5× on lane 0 (RMSNorm). Only lane 0 has an ln unit; lanes 1..N−1 only bypass |
| `ln_pre_scaled`| input     | 1                                | 0: input has `FRAC_W` fractional bits (shifted up by `LN_EXTRA`); 1: already `FRAC_W+LN_EXTRA` (RMSNorm mean) |
| `vec_in`       | input     | `[WIDTH_IN-1:0][N]`              | Input vector                                    |
| `log_out`      | output    | `[WIDTH_OUT-1:0][N]`             | `tr_ln_alu`'s (post-conditioned) log output     |
| `a_idx_out`    | output    | `[LUT_IDX_W-1:0][N]`             | Shared-LUT anchor index for `shared_lut_rom`    |
| `mantisa_out`  | output    | `[WIDTH_OUT-1:0][N]`             | Taylor-mantissa term to multiply against `e_a`  |
| `is_zero_out`  | output    | `[N]`                            | Whether this lane's `exp` result should be 0    |

---

## `tr_ln_alu`

**File:** [rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_ln_alu.sv](../../rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_ln_alu.sv)

### Operation

Integer natural-logarithm approximator: MSB exponent + **piecewise-linear fit of `ln(1+f)`** over the mantissa fraction. Finds `xq`'s MSB (`floor(log2(xq))`) using a balanced binary-search tree (not a linear scan — this was a real critical-path fix, see the FPGA M2 timing results), normalizes `xq` so that bit sits at the top of the word, and takes the `BITS` bits below it as the mantissa fraction `fk`. The mantissa range is split into `SEGMENTS` equal segments (top `log2(SEGMENTS)` bits of `fk`), each with its own least-squares line, constants quantized to `COEF_BITS`: `aq = msb − BITS`, `yq = (aq·L·2^BITS + A[seg]·fk + B[seg]·2^BITS + 2^(COEF_BITS−1)) >>> COEF_BITS` with `L = round(ln2·2^COEF_BITS)`. All multiplies are by constants (shift-add trees, no general multiplier); the result is saturated to `OUT_WIDTH`. `xq==0` short-circuits to `yq=0`. Bit-exact to `tools/models/vpu_ops.py::_ln_core` and to `tr_pwl_ln_scalar` (verified exhaustively over the input width for 10 configurations via `verify_block.py ln` with `LN_CFG`). Supported constants: `SEGMENTS`∈{1,2,4} at `COEF_BITS=8`, {1,2} at 6, 2 at 10 (an unsupported pair stops elaboration).

### Parameters

| Parameter   | Default | Description                                    |
|-------------|---------|--------------------------------------------------|
| `WIDTH`     | `16`    | Input width (`xq`)                              |
| `BITS`      | `4`     | Fractional bits of `xq` (the shared unit uses 6)  |
| `OUT_WIDTH` | `8`     | Output width (`yq`), saturated                   |
| `SEGMENTS`  | `2`     | Piecewise-linear segments (1, 2, 4)              |
| `COEF_BITS` | `8`     | Fixed-point bits of the segment constants (6, 8, 10) |

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
