# MX Modules — `rtl_soc`

These modules live in [rtl_soc/mx_modules/](../../rtl_soc/mx_modules/). They support the MX (Microscaling shared-exponent) datapath variant (`tr_soc_top_mx`) — see [mx_format_operation.md](../mx_format_operation.md) for the full dataflow these two modules sit in. `requantize_engine_mx` (the MX counterpart of `requantize_engine_int`) lives in [requantize_engine.md](requantize_engine.md) instead, alongside its INT sibling.

---

## `dynamic_shifter_mx`

**File:** [rtl_soc/mx_modules/dynamic_shifter_mx.sv](../../rtl_soc/mx_modules/dynamic_shifter_mx.sv)

### Operation

N-wide bidirectional arithmetic shifter for MX shared-exponent scaling. Combinationally shifts every lane of `data_in` by `shift_amount` (the MX block's shared exponent): left (`shift_dir=1`) to expand a narrow MX mantissa up before the VPU consumes it, or right (`shift_dir=0`, sign-preserving) to compress a wide VPU result back down when the MX formatter re-quantizes it. Each lane is cast to `OUT_W` before shifting so left-shifted bits have room to expand without truncation.

### Parameters

| Parameter | Default | Description                                                          |
|-----------|---------|----------------------------------------------------------------------|
| `N`       | `4`     | Number of parallel lanes                                              |
| `IN_W`    | `8`     | Input data width (`data_in`)                                          |
| `OUT_W`   | `16`    | Output data width (`data_out`); must be `>= IN_W`                     |

### Ports

| Port           | Direction | Width               | Description                                    |
|----------------|-----------|---------------------|-------------------------------------------------|
| `data_in`      | input     | `[IN_W-1:0][N]`      | Signed input vector                             |
| `shift_amount` | input     | 8                    | The shared exponent, signed                     |
| `shift_dir`    | input     | 1                    | 1: left (expand), 0: right (compress)           |
| `data_out`     | output    | `[OUT_W-1:0][N]`     | Shifted result vector                           |

---

## `formatter_mx`

**File:** [rtl_soc/mx_modules/formatter_mx.sv](../../rtl_soc/mx_modules/formatter_mx.sv)

### Operation

Converts an N-wide VPU output vector into MX (shared-exponent) format: N narrow mantissas plus one shared exponent. Finds the maximum magnitude across the N lanes, derives the minimum right-shift needed to fit that maximum into `MX_W`-bit mantissas (via a priority-encoder bit-width finder), then right-shifts (truncating) every lane by that shared amount and registers the mantissas plus the shared exponent together. This is the final compression stage before a VPU result goes back to SRAM.

### Parameters

| Parameter | Default | Description                                        |
|-----------|---------|---------------------------------------------------------|
| `N`       | `4`     | Number of parallel lanes                                  |
| `VPU_W`   | `16`    | Width of the incoming per-lane VPU data (`vpu_data_in`)    |
| `MX_W`    | `8`     | Width of the outgoing per-lane mantissa (`mx_mantissas`)   |

### Ports

| Port            | Direction | Width               | Description                            |
|-----------------|-----------|---------------------|-------------------------------------------|
| `clk`, `rst_n`  | input     | 1                    | Clock / active-low reset                  |
| `valid_in`      | input     | 1                    | Input valid                                |
| `vpu_data_in`   | input     | `[VPU_W-1:0][N]`     | Wide VPU result vector                     |
| `valid_out`     | output    | 1                    | Output valid                               |
| `mx_mantissas`  | output    | `[MX_W-1:0][N]`      | Compressed per-lane mantissas              |
| `mx_shared_exp` | output    | 8                    | Shared exponent for the compressed vector  |
