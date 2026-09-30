# Architecture Overview

This branch (`tr-vpu_rtl`) contains the RTL for a Taylor-Region (TR) Vector Processing Unit: a shared, software-defined log/exp datapath that implements GELU, Softmax, and RMSNorm on top of one reusable crossbar, plus a linear (matmul) engine that feeds it. See [tr_soc_architecture.drawio](tr_soc_architecture.drawio) for the full block diagram.

## The core idea: one shared nonlinear crossbar

`tr_nonlinear_vpu` ([rtl_soc/tr_nonlinear_vpu/tr_nonlinear_vpu.sv](../rtl_soc/tr_nonlinear_vpu/tr_nonlinear_vpu.sv)) is a purely combinational/pipelined crossbar with **zero internal state**. It routes SRAM data through isolated math modifiers (`alpha_stabilizer`, `scalar_sub`, `symmetry_modifier`, `piped_max` — see [module_reference/peripheral_modules.md](module_reference/peripheral_modules.md)), a shared log-domain backbone (`tr_backbone_wrapper` — see [module_reference/tr_backbone.md](module_reference/tr_backbone.md)), and local vector/MAC engines ([module_reference/mult_engines.md](module_reference/mult_engines.md)).

Because it has no state of its own, a single GELU, Softmax, or RMSNorm operation isn't one shot through the crossbar — it's a **sequence of passes**, each reconfiguring the crossbar's mux selects/enables and reading back the previous pass's result. That sequencing is owned entirely by the controller, not the VPU itself:

- **`tr_soc_ctrl_int`** / **`tr_soc_ctrl_mx`** ([module_reference/tr_soc_ctrl.md](module_reference/tr_soc_ctrl.md)) are the MMIO-driven master FSMs. An MMIO command write (GELU/Softmax/RMSNorm/matmul) kicks off a fixed state sequence (`GL_P1..GL_P3`, `SM_P1..SM_P4`, `RM_P1..RM_P4`) that drives the crossbar pass-by-pass, gated on each pass's own valid handshake, holding intermediate per-lane results in scratch registers between passes.

This shared-crossbar design is the central area/energy trade-off of the whole project: one physical netlist does all three nonlinear ops (plus the log/exp math both matmul-adjacent RMSNorm and Softmax need), at the cost of multi-pass latency per operation instead of a single-shot dedicated pipeline. See `ENERGY_SUMMARY_Q4.4.md` (on the synthesis branch) for where that trade actually pays off and where it doesn't.

## Two datapath variants: INT vs MX

The repo ships two parallel top-level integrations that differ only in how they rescale data between the linear and nonlinear stages:

| | `tr_soc_top_int` | `tr_soc_top_mx` |
|---|---|---|
| Controller | `tr_soc_ctrl_int` | `tr_soc_ctrl_mx` |
| Rescaling | `requantize_engine_int`: explicit fixed-point `Acc·M>>>S` requantizer | `requantize_engine_mx` + `dynamic_shifter_mx` + `formatter_mx`: shared-exponent (Microscaling) format |
| VPU operating width | narrow (`W_VEC`) with a separate wider internal backbone (`W_MAC`) | runs at one wide `VPU_W` throughout (no separate narrow SRAM format) |

Both instantiate the exact same `tr_nonlinear_vpu` and `dot_product_engine` — only the glue between linear and nonlinear stages changes. See [mx_format_operation.md](mx_format_operation.md) for the full MX dataflow (matmul → requantize → dynamic-shift-expand → VPU → formatter-compress).

## The linear side

`dot_product_engine` ([module_reference/mult_engines.md](module_reference/mult_engines.md) covers its underlying `mac_array_engine` lanes) computes `Out[i] = Σ(A[i]·B) + C[i]` — M parallel `mac_array_engine` lanes sharing one broadcast activation vector. `tr_matmul_ctrl` ([module_reference/tr_soc_ctrl.md](module_reference/tr_soc_ctrl.md)) sequences it into a full tiled matmul; the same sequencer is reused by both the INT and MX tops since it only drives handshakes, not data.

## Baseline comparisons

`rtl_baseline/` holds two families of standalone RTL kept purely for comparison, **not** instantiated by either SoC top:

- `tr_baseline/` — standalone `tr_gelu`/`tr_softmax`/`tr_rmsnorm`/`tr_swiglu` modules built from the same shared backbone primitives, but wired directly rather than through the SoC controller/crossbar. Useful for isolating the backbone's own behavior, or (for `tr_rmsnorm`) numerically identical to the production RM_P1..RM_P4 sequence — see [verification.md](verification.md). Shares one helper module, `vec_mul` (see [module_reference/rtl_baseline.md](module_reference/rtl_baseline.md)).
- `ibert_baseline/` — an independent I-BERT-style implementation (iterative divider-based) used as the paper's baseline comparison point for energy/area.

## Where to go next

- [module_reference/](module_reference/) — per-block port/parameter reference, one file per `rtl_soc/` subfolder.
- [verification.md](verification.md) — how to actually run any of this (`verify_block.py`, the `*_energy_tb.sv` convention).
- [mx_format_operation.md](mx_format_operation.md) — the MX datapath in detail.
