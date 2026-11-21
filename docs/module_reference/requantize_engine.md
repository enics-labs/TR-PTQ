# Requantize Engine — `rtl_soc`

This module lives in [rtl_soc/requantize_engine/](../../rtl_soc/requantize_engine/). It compresses a wide accumulator back down to a narrow output for the nonlinear VPU / SRAM to consume, in the fixed-point (INT) datapath of `tr_soc_top_int`.

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
