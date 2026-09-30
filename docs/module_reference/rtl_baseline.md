# Shared Support — `rtl_baseline`

`rtl_baseline/tr_baseline/` and `rtl_baseline/ibert_baseline/` mostly hold standalone, self-contained blocks (`tr_gelu`, `tr_softmax`, `tr_rmsnorm`, `tr_swiglu`, `ibert_*`) documented by [architecture.md](../architecture.md) and [verification.md](../verification.md) rather than here. The one exception is `vec_mul`, a shared helper used by every `tr_baseline/` block.

---

## `vec_mul`

**File:** [rtl_baseline/tr_baseline/vec_mul.sv](../../rtl_baseline/tr_baseline/vec_mul.sv)

### Operation

A 4-stage pipelined vector multiplier supporting both dot-product and element-wise modes, used only by the standalone `tr_baseline/` blocks (`tr_gelu`, `tr_rmsnorm`, `tr_softmax`, `tr_swiglu`) — the production `tr_nonlinear_vpu` crossbar uses `vec_mul_array_engine` (see [mult_engines.md](mult_engines.md)) instead. The pipeline stages are:

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
|-----------|---------|-------------------------------------------|
| `N`       | `16`    | Number of vector lanes                   |
| `W`       | `8`     | Input operand bit width                  |
| `ACC_W`   | `32`    | Accumulator / output bit width           |

### Ports

| Port             | Direction | Width              | Description                                              |
|------------------|-----------|--------------------|------------------------------------------------------------|
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
