# Requantize Engine — `rtl_soc`

These modules live in [rtl_soc/requantize_engine/](../../rtl_soc/requantize_engine/). Both compress a wide accumulator back down to a narrow output for the nonlinear VPU / SRAM to consume — `requantize_engine_int` for the fixed-point (INT) datapath (`tr_soc_top_int`), `requantize_engine_mx` for the shared-exponent (MX) datapath (`tr_soc_top_mx`; see [mx_format_operation.md](../mx_format_operation.md)).

---

## `requantize_engine_int`

**File:** [rtl_soc/requantize_engine/requantize_engine_int.sv](../../rtl_soc/requantize_engine/requantize_engine_int.sv)

### Operation

Vectorized integer requantizer with saturation: `Out = Saturate((Acc * M + Bias) >>> S)`. 3-stage pipeline for high-frequency timing closure: (1) wide signed multiply (`acc_in * multiplier`), (2) add a rounding bias of `1<<<(shift-1)` then arithmetic-right-shift by the controller-supplied `shift` amount, (3) saturate to the signed `OUT_W` range. This is the "requantize" step of matmul → requantize → nonlinear-VPU in the INT datapath.

### Parameters

| Parameter | Default | Description                                                    |
|-----------|---------|------------------------------------------------------------------|
| `N`       | `16`    | Number of parallel lanes                                          |
| `ACC_W`   | `32`    | Input accumulator width (`acc_in`)                                 |
| `MUL_W`   | `32`    | Multiplier scale width (`multiplier`, "M" from the controller)     |
| `SHIFT_W` | `6`     | Shift amount width (`shift`, "S" from the controller)              |
| `OUT_W`   | `8`     | Target output width (`out_vec`)                                    |

### Ports

| Port         | Direction | Width            | Description                                |
|--------------|-----------|-------------------|------------------------------------------------|
| `clk`, `rst_n` | input   | 1                  | Clock / active-low reset                        |
| `in_valid`   | input     | 1                  | Input valid                                      |
| `in_ready`   | output    | 1                  | Ready to accept a new beat                       |
| `acc_in`     | input     | `[ACC_W-1:0][N]`   | Wide accumulator vector                          |
| `multiplier` | input     | `MUL_W`            | Scale multiplier "M", from the controller        |
| `shift`      | input     | `SHIFT_W`          | Right-shift amount "S", from the controller      |
| `out_valid`  | output    | 1                  | Output valid                                      |
| `out_ready`  | input     | 1                  | Downstream ready                                  |
| `out_vec`    | output    | `[OUT_W-1:0][N]`   | Saturated, requantized result vector             |

---

## `requantize_engine_mx`

**File:** [rtl_soc/requantize_engine/requantize_engine_mx.sv](../../rtl_soc/requantize_engine/requantize_engine_mx.sv)

### Operation

2-stage pipelined MX-format requantizer: compresses N wide accumulator outputs down to `OUT_W`-bit mantissas plus a combined exponent. Stage 1 finds the max magnitude across the N `dot_in` lanes (combinational max-tree) and latches the base exponent (`exp_act_in + exp_weight_in`, the product of the two MX inputs' shared exponents that produced this accumulation). Stage 2 leading-zero-counts the registered max to derive the minimum right-shift that fits it into `OUT_W-1` magnitude bits, shifts every lane by that shared amount, and adds the shift onto the base exponent to produce `exp_total_out` — the MX-format counterpart of `requantize_engine_int`'s fixed-point `Mult>>>Shift` requantization.

### Parameters

| Parameter | Default | Description                                              |
|-----------|---------|--------------------------------------------------------------|
| `N`       | `4`     | Number of parallel lanes                                       |
| `ACC_W`   | `32`    | Width of the incoming per-lane accumulator value (`dot_in`)     |
| `OUT_W`   | `8`     | Width of the outgoing per-lane mantissa (`req_vec_out`)         |

### Ports

| Port             | Direction | Width               | Description                                          |
|------------------|-----------|---------------------|----------------------------------------------------------|
| `clk`, `rst_n`   | input     | 1                    | Clock / active-low reset                                  |
| `dot_in_valid`   | input     | 1                    | Input valid                                                |
| `req_out_valid`  | output    | 1                    | Output valid                                               |
| `dot_in`         | input     | `[ACC_W-1:0][N]`     | Wide accumulator vector                                    |
| `exp_act_in`     | input     | 8                    | Activation operand's shared exponent                       |
| `exp_weight_in`  | input     | 8                    | Weight operand's shared exponent                            |
| `req_vec_out`    | output    | `[OUT_W-1:0][N]`     | Compressed per-lane mantissas                               |
| `exp_total_out`  | output    | 8                    | Combined output exponent (base + derived shift)             |
