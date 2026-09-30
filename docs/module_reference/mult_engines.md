# Mult Engines — `rtl_soc`

These modules live in [rtl_soc/mult_engines/](../../rtl_soc/mult_engines/). Both are streaming-pipelined, N-wide parallel-lane multiply engines shared across the linear (matmul) and nonlinear (`tr_nonlinear_vpu`) datapaths — `mac_array_engine` reduces to a scalar dot product, `vec_mul_array_engine` keeps per-lane elementwise products.

---

## `mac_array_engine`

**File:** [rtl_soc/mult_engines/mac_array_engine.sv](../../rtl_soc/mult_engines/mac_array_engine.sv)

### Operation

Evaluates `out_dot = SUM(a[i]*b[i]) + C` over a 4-stage pipeline: (1) register inputs, (2) compute the N per-lane products (mode-selectable signed/unsigned operands), (3) reduce the N products with a combinational adder tree, (4) either initialize the accumulator with the reduced sum plus the bias `C` (`clear_acc=1`, first beat of a K-dimension reduction) or add the reduced sum onto the running `out_dot` (`clear_acc=0`, subsequent beats). Used for matmul dot products (via `dot_product_engine`) and for the VPU's variance/softmax-sum reductions.

### Parameters

| Parameter | Default | Description                                               |
|-----------|---------|-----------------------------------------------------------|
| `N`       | `16`    | Number of parallel lanes                                    |
| `W`       | `8`     | Input operand width (`a`/`b`)                                |
| `ACC_W`   | `32`    | Accumulator/output width (also bias `c` and `out_dot`)       |

### Ports

| Port        | Direction | Width           | Description                                        |
|-------------|-----------|------------------|-------------------------------------------------------|
| `clk`, `rst_n` | input  | 1                | Clock / active-low reset                              |
| `in_valid`  | input     | 1                | Input valid                                            |
| `in_ready`  | output    | 1                | Ready to accept a new beat                             |
| `op_mode`   | input     | 2                | 0: SS, 1: SU, 2: UU (signed/unsigned operand select)   |
| `a`, `b`    | input     | `[W-1:0][N]`     | Per-lane operand vectors                               |
| `c`         | input     | `ACC_W`          | Scalar bias, added on `clear_acc`                      |
| `clear_acc` | input     | 1                | 1: initialize accumulator with `C`; 0: accumulate       |
| `out_valid` | output    | 1                | Output valid                                            |
| `out_ready` | input     | 1                | Downstream ready                                        |
| `out_dot`   | output    | `ACC_W`          | Running/final dot-product result                        |

---

## `vec_mul_array_engine`

**File:** [rtl_soc/mult_engines/vec_mul_array_engine.sv](../../rtl_soc/mult_engines/vec_mul_array_engine.sv)

### Operation

Evaluates `out_vec[i] = a[i]*b[i]` for each of the N lanes over a 3-stage pipeline: (1) register inputs, (2) compute the N per-lane products (mode-selectable signed/unsigned operands, same convention as `mac_array_engine`), (3) register the products, sign-extended/cast to the output width. Used for Softmax probability scaling and GELU's final elementwise gate — this is `tr_nonlinear_vpu`'s crossbar "vecmul" engine.

### Parameters

| Parameter | Default | Description                        |
|-----------|---------|-----------------------------------------|
| `N`       | `16`    | Number of parallel lanes                 |
| `W`       | `8`     | Input operand width (`a`/`b`)             |
| `ACC_W`   | `32`    | Output width (`out_vec`)                  |

### Ports

| Port        | Direction | Width           | Description                                       |
|-------------|-----------|------------------|--------------------------------------------------------|
| `clk`, `rst_n` | input  | 1                | Clock / active-low reset                               |
| `in_valid`  | input     | 1                | Input valid                                             |
| `in_ready`  | output    | 1                | Ready to accept a new beat                              |
| `op_mode`   | input     | 2                | 0: SS, 1: SU, 2: UU (signed/unsigned operand select)    |
| `a`, `b`    | input     | `[W-1:0][N]`     | Per-lane operand vectors                                |
| `out_valid` | output    | 1                | Output valid                                             |
| `out_ready` | input     | 1                | Downstream ready                                         |
| `out_vec`   | output    | `[ACC_W-1:0][N]` | Per-lane product results                                 |
