# Peripheral Modules — `tr_nonlinear_vpu`

These modules live in [rtl_soc/tr_nonlinear_vpu/peripheral_modules/](../rtl_soc/tr_nonlinear_vpu/peripheral_modules/).

---

## `piped_max`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/piped_max.sv](../rtl_soc/tr_nonlinear_vpu/peripheral_modules/piped_max.sv)

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

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/scalar_sub.sv](../rtl_soc/tr_nonlinear_vpu/peripheral_modules/scalar_sub.sv)

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

## `vec_mul`

**File:** [rtl_soc/tr_nonlinear_vpu/peripheral_modules/vec_mul.sv](../rtl_soc/tr_nonlinear_vpu/peripheral_modules/vec_mul.sv)

### Operation

A 4-stage pipelined vector multiplier supporting both dot-product and element-wise modes. The pipeline stages are:

| Stage | Action                                                               |
|-------|----------------------------------------------------------------------|
| 1     | Register inputs (`a`, `b`, control signals)                         |
| 2     | Compute per-lane products with configurable sign semantics           |
| 3     | Sum all products (DOT mode) or pass per-lane products through (ELEMWISE mode) |
| 4     | Write output; accumulate into `out_vec[0]` (DOT) or broadcast lane results (ELEMWISE) |

**Sign modes** (`op_mode`):

| `op_mode` | Mode | Multiplication                            |
|-----------|------|-------------------------------------------|
| `2'd0`    | SS   | `$signed(a[i]) * $signed(b[i])`           |
| `2'd1`    | SU   | `$signed(a[i]) * $signed({1'b0, b[i]})`   |
| `2'd2`    | UU   | `$signed({1'b0,a[i]}) * $signed({1'b0,b[i]})` |

**Execution modes** (`mode_elemwise`):

| `mode_elemwise` | Mode     | `out_vec`                          | `out_valid_mask`    |
|-----------------|----------|------------------------------------|---------------------|
| `0`             | DOT      | `out_vec[0]` = accumulated sum; others = 0 | bit 0 only      |
| `1`             | ELEMWISE | `out_vec[i]` = `a[i] * b[i]`      | all N bits set      |

In DOT mode, successive results accumulate into `out_vec[0]` across beats unless `clear_acc` is asserted, which resets the accumulator to the current beat's partial sum.

Flow control is a simple valid/ready handshake. The pipeline advances only when the output is not stalled (`advance = ~out_valid || (out_valid && out_ready)`).

### Parameters

| Parameter | Default | Description                              |
|-----------|---------|------------------------------------------|
| `N`       | `16`    | Number of vector lanes                   |
| `W`       | `8`     | Input operand bit width                  |
| `ACC_W`   | `32`    | Accumulator / output bit width           |

### Ports

| Port             | Direction | Width              | Description                                              |
|------------------|-----------|--------------------|----------------------------------------------------------|
| `clk`            | input     | 1                  | Clock                                                    |
| `rst_n`          | input     | 1                  | Active-low synchronous reset                             |
| `in_valid`       | input     | 1                  | Input handshake valid                                    |
| `in_ready`       | output    | 1                  | Input handshake ready (back-pressure)                    |
| `op_mode`        | input     | 2                  | Sign mode: `0`=SS, `1`=SU, `2`=UU                       |
| `mode_elemwise`  | input     | 1                  | `0`=DOT product, `1`=element-wise                        |
| `a`              | input     | `[W-1:0][N]`       | First operand vector (raw bits)                          |
| `b`              | input     | `[W-1:0][N]`       | Second operand vector (raw bits)                         |
| `clear_acc`      | input     | 1                  | When set, resets the DOT accumulator on the current beat |
| `out_valid`      | output    | 1                  | Output handshake valid                                   |
| `out_ready`      | input     | 1                  | Output handshake ready (consumer back-pressure)          |
| `out_valid_mask` | output    | N                  | Bitmask indicating which `out_vec` lanes hold valid data |
| `out_vec`        | output    | `[ACC_W-1:0][N]`   | Result vector (signed)                                   |
