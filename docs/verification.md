# Verification

## Running a testbench directly

Every block has an `.f` file (an `xrun` file list: global sim config + RTL sources + one `_tb.sv`). Run it from `workspace/`:

```sh
cd workspace
xrun -f ../tb_soc/<block>/<block>.f
```

Self-checking testbenches (e.g. `mac_array_engine_tb.sv`) print `PASS`/`FAIL` directly. File-I/O testbenches (see below) instead need the golden-vector flow.

## The golden-vector flow: `verify_block.py`

Most blocks are verified against a C++ golden model (`emulation/cpu_math_model.cpp`, backed by the bit-true reference functions in `emulation/tr_math_model.hpp`) rather than hand-picked directed vectors. Run from `workspace/`:

```sh
cd workspace
python3 ../verification/verify_block.py <block_name>
```

This does 4 things: (1) compiles `cpu_math_model.cpp`, (2) runs it in the given mode to generate `inputs.txt`/`expected.txt` (random vectors, seeded), (3) runs the block's `.f` file via `xrun`, which reads `inputs.txt` and writes `hdl_out.txt`, (4) diffs `expected.txt` against `hdl_out.txt` element-by-element against the block's `max_error` tolerance.

**The block name is passed straight through to the C++ model as its mode string** (`get_config()`'s dict key doubles as `argv[1]` to `cpu_model`) — if you add a new block name in `verify_block.py`, `cpu_math_model.cpp` needs a matching `mode == "..."` branch, or the run fails at step 3 with an "Unknown mode" error from the C++ side, not a Python error.

### Block name reference

| Block | Tests | Golden model source |
|---|---|---|
| `exp` | `tr_exp_alu` in isolation | `tr_approx_exp_scalar_2rd` |
| `ln` | `tr_ln_alu` in isolation | `tr_new_ln_scalar` |
| `gelu` | Production GL_P1..GL_P3 via `tr_soc_top_int` (CMD=0x02) | `gelu_hw_model` |
| `gelu_baseline` | Standalone `tr_gelu.sv` (`rtl_baseline/`) | Same `gelu_hw_model` — identical algorithm to production |
| `gelu_fused` | Real firmware composition: raw accumulator → requantize → GELU | `gelu_hw_scalar(quant_model(...))` |
| `softmax` | Production SM_P1..SM_P4 via `tr_soc_top_int` (CMD=0x01) | `softmax_hw_model` (log-sum-exp) |
| `softmax_baseline` | Standalone `tr_softmax.sv` (`rtl_baseline/`) | **Different** algorithm (explicit reciprocal-via-exp) — its own dedicated golden model |
| `rmsnorm` | Production RM_P1..RM_P4 via `tr_soc_top_int` (CMD=0x03) | `rmsnorm_hw_model` |
| `rmsnorm_baseline` | Standalone `tr_rmsnorm.sv` (`rtl_baseline/`) | Same `rmsnorm_hw_model` — identical, sign-guard included |
| `swiglu` | Standalone `tr_swiglu.sv` (`rtl_baseline/`) | Inline model in `cpu_math_model.cpp` |
| `quant` | `requantize_engine_int` in isolation | `quant_model` |
| `matmul` | `dot_product_engine` + `requantize_engine_int` | `matmul_model` |

Only `softmax_baseline` needed a bespoke golden model — `tr_softmax.sv` is a genuinely different algorithm from the production log-sum-exp path (own "is_zero→255" anchor vs. the crossbar's "is_zero→128-doubling" trick). `gelu_baseline`/`rmsnorm_baseline` reuse their production counterparts' models directly since the standalone modules implement the identical algorithm, just wired without the SoC controller.

**Known gap:** `rmsnorm_hw_model` (used by both `rmsnorm` and `rmsnorm_baseline`) does not implement the sign-guard fix that the real RTL has (see `rmsnorm_hw_model_signguard_fixed` in `tr_math_model.hpp`, currently unused dead code). Random `[-128,127]` stimulus rarely triggers the affected regime (small `Σx²`), so this has likely never surfaced as a test failure, but it means the golden model can disagree with correct RTL specifically in that regime.

## Energy characterization: `*_energy_tb.sv`

Separate from correctness verification. Each of these instruments a **new, dedicated** testbench (never the correctness `_tb.sv`) that drives one representative 8-lane stimulus vector, counts clock cycles from stimulus-valid to result-valid, and captures a VCD over exactly that window:

```
tb_soc/tr_nonlinear_vpu/tr_nonlinear_vpu_{gelu,softmax,rmsnorm}_energy.f
tb_baseline/tr_baseline/tr_{gelu,softmax,rmsnorm}/tr_*_energy.f
tb_baseline/ibert_baseline/ibert_{gelu,softmax,rmsnorm}/ibert_*_energy.f
```

The VCD feeds Genus (65nm synthesis, on the `synthesis` branch family) via `read_vcd -vcd_scope <tb>/<dut> <file>.vcd` against the already-synthesized netlist, then `report_power`. `Energy = Total_Power(µW) × N_cycles × T_clock(ns)`. All nine designs above use the **same** representative stimulus vector for direct comparability. See `ENERGY_SUMMARY_Q4.4.md` (synthesis branch) for the resulting numbers.

`tr_nonlinear_vpu`'s own energy pilots are the trickiest of the nine: it has no single valid_in/valid_out handshake (a crossbar with 4 separate valid outputs), so each pilot replicates the controller's exact multi-pass sequence directly against the VPU, including two testbench pitfalls worth knowing about if you're writing a new one:

- **Same-engine back-to-back passes race.** If one pass's engine call (e.g. `vecmul_pass()`) immediately follows another pass on the *same* engine with no different-engine pass in between, the crossbar's stale prior-pass output gets sampled. Needs a "settled" task variant (`#1` inside the wait loop) plus a resync clock edge before the next pulse.
- **Reading a register right after `@(posedge clk)` catches its pre-update value** (an NBA/Active-region ordering artifact) — always add a settled read (`@(posedge clk); #1;`) before capturing a result that was reached via `while(!valid) @(posedge clk)`.

Both are documented inline in the affected `*_energy_tb.sv` files where they apply.
