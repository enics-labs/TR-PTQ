# Architecture

This branch (`tr-vpu_rtl`) contains the RTL of a Taylor-Region (TR) vector processing unit: a shared, software-defined
log/exp datapath that implements the nonlinear operators of INT8 transformer inference on one reusable crossbar, plus
the linear (matmul) engine that feeds it. Block diagram: [tr_soc_architecture.drawio](tr_soc_architecture.drawio).

## One shared nonlinear crossbar

`tr_nonlinear_vpu` ([rtl_soc/tr_nonlinear_vpu/tr_nonlinear_vpu.sv](../rtl_soc/tr_nonlinear_vpu/tr_nonlinear_vpu.sv))
is a stateless, N=8 lane crossbar. It routes SRAM data through small math modifiers, a shared log-domain backbone and
local vector/MAC engines:

| Block | Role | Reference |
|---|---|---|
| `tr_backbone_wrapper` | pre-conditioner, shared `tr_ln_alu`, post-conditioner, `tr_exp_alu` | [tr_backbone.md](module_reference/tr_backbone.md) |
| `mac_array_engine` | sum of squares, softmax sum | [mult_engines.md](module_reference/mult_engines.md) |
| `vec_mul_array_engine` + `vecmul_out_cast` | element-wise products and their scaling / rounding / saturation | [mult_engines.md](module_reference/mult_engines.md) |
| `piped_max` | softmax row max and SwiGLU maximum of `|silu * up|` | [peripheral_modules.md](module_reference/peripheral_modules.md) |
| `sigma_rom`, `recip_rom` | SwiGLU sigmoid table, RMSNorm growth-branch reciprocal | [peripheral_modules.md](module_reference/peripheral_modules.md) |
| `scalar_sub`, `alpha_stabilizer`, `symmetry_modifier` | softmax `x - max`; GELU only | [peripheral_modules.md](module_reference/peripheral_modules.md) |

One `tr_ln_alu` (piecewise-linear, BITS = 6) serves every operator that needs a logarithm.

Because the crossbar has no state, an operation is a sequence of passes. The sequencing is owned by the controller
`tr_soc_ctrl_int` ([tr_soc_ctrl.md](module_reference/tr_soc_ctrl.md)): an MMIO command write starts a fixed state
sequence that drives the crossbar pass by pass, gated on each pass's own valid handshake.

## Commands

| Command | Operation | Granularity |
|---|---|---|
| `0x01` | softmax tile with `z = max + ln S` readout | one 8-lane tile |
| `0x02` | GELU (GELU build only) | one 8-lane vector |
| `0x04` | streaming matmul | tile counts in registers |
| `0x07` / `0x08` | RMSNorm sum-of-squares, first tile / next tile | one tile |
| `0x09` | RMSNorm scale (mean, ln, exp, `inv_rms` register) | once per row |
| `0x0A` | RMSNorm apply | one tile |
| `0x05` | SwiGLU silu group | one 8-lane group |
| `0x06` | SwiGLU running maximum of `|silu * up|` | one group |
| `0x0B` | SwiGLU row gain | once per row |
| `0x0C` | SwiGLU apply | one group |

RMSNorm and SwiGLU rows are longer than 8 elements, so the host sequences them over tiles or groups. Register map and
state sequences: [tr_soc_ctrl.md](module_reference/tr_soc_ctrl.md). Numerical specification and accuracy study of the
Llama configuration live in the integration repository (`docs/llama/`).

## Build flavors

`GELU_EN` is a compile-time parameter carried by `tr_soc_top_int`, `tr_soc_ctrl_int`, `tr_nonlinear_vpu`,
`tr_backbone_wrapper` and `vecmul_out_cast`. Its default comes from the macro `GELU_ENABLE`.

| Build | `GELU_EN` | Content |
|---|---|---|
| default (Llama) | 0 | no GELU; command `0x02` is ignored |
| GELU | 1 | adds GELU support for ViT-style models |

What the default build removes, and why it is safe: softmax and RMSNorm read the ln result of lane 0 only, and every
other use of the backbone bypasses the ln. So the build drops the ln unit on lanes 1 to 7, `pre_ln_modifier` (its
"+1.0" mode is GELU-only), the post-ln modes other than bypass and the rounded `-0.5x` on lane 0, `alpha_stabilizer`,
`symmetry_modifier`, `scratch_b`, the vec-mul cast modes `001` and `010`, and the GELU controls.

## Timing structure

The backbone is a three-register-stage pipeline: input registers; `pre_ln` and `tr_ln_alu`; round-back, saturation
and `post_ln`; `tr_exp_alu` combinational to the outputs (three cycles from `in_valid` to `out_valid`). The register
after the ln closes the 3.3 ns clock. Rounding in the wide shifters is written shifter-first
(`((2x >>> s) + 1) >>> 1`), which removes a subtract and an adder from the path and gives identical results.

## INT and MX variants

| | `tr_soc_top_int` | `tr_soc_top_mx` |
|---|---|---|
| Controller | `tr_soc_ctrl_int` | `tr_soc_ctrl_mx` |
| Rescaling | `requantize_engine_int`: explicit `acc * M >>> S` | `requantize_engine_mx` + `dynamic_shifter_mx` + `formatter_mx`: shared exponent |
| VPU width | narrow (`W_VEC`) with a wider internal backbone | one wide `VPU_W` throughout |

Both use the same `tr_nonlinear_vpu` and `dot_product_engine`. The MX variant has not been brought up to the command
set above. See [mx_format_operation.md](mx_format_operation.md).

## Linear side

`dot_product_engine` computes `Out[i] = sum(A[i] * B) + C[i]` with M parallel `mac_array_engine` lanes sharing one
broadcast vector. `tr_matmul_ctrl` sequences it into a tiled matmul and is shared by both tops.

## Baselines

`rtl_baseline/` holds standalone RTL for comparison, not instantiated by either top: `tr_baseline/` (dedicated TR
modules built from the same primitives) and `ibert_baseline/` (an I-BERT-style, divider-based implementation).
See [module_reference/rtl_baseline.md](module_reference/rtl_baseline.md).
