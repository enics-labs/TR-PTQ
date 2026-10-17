# SoC Controllers — `rtl_soc`

These modules live in [rtl_soc/tr_soc_ctrl/](../../rtl_soc/tr_soc_ctrl/) (plus [tr_matmul_ctrl.sv](../../rtl_soc/tr_matmul_ctrl.sv) at the `rtl_soc/` root, included here since it's driven by the same MMIO dispatch). They are the MMIO-driven master FSMs that sequence `tr_nonlinear_vpu`'s stateless crossbar pass-by-pass into a GELU/Softmax/RMSNorm operation, or kick off a streaming matmul.

---

## `tr_soc_ctrl_int`

**File:** [rtl_soc/tr_soc_ctrl/tr_soc_ctrl_int.sv](../../rtl_soc/tr_soc_ctrl/tr_soc_ctrl_int.sv)

### Operation

On an MMIO command write, sequences `tr_nonlinear_vpu`'s crossbar through the multi-pass GELU/Softmax/RMSNorm routes, or kicks off a streaming matmul via `tr_matmul_ctrl`.

**MMIO register map:**

| Address | Name              | Meaning                                                      |
|---------|-------------------|---------------------------------------------------------------|
| `0x00`  | `ADDR_CMD`        | Write triggers a command (table below)                          |
| `0x04`  | `ADDR_STATUS`     | `{done, busy}`                                                 |
| `0x08`  | `ADDR_REQ_MULT`   | Requantizer multiplier "M"                                     |
| `0x0C`  | `ADDR_REQ_SHIFT`  | Requantizer shift "S"                                          |
| `0x10`  | `ADDR_MM_ROWS`    | Streaming-matmul row-tile count                                 |
| `0x14`  | `ADDR_MM_CTILES`  | Streaming-matmul contraction-tile count                         |
| `0x18`  | `ADDR_RMS_LOG2N`  | RMSNorm: log2 of the row length (read by command 9)             |
| `0x1C`  | `ADDR_SW_E`       | SwiGLU: row gain `e` (signed, read-only; valid after command 0xB) |
| `0x20`  | `ADDR_SM_Z`       | Softmax: tile `z = sat8(max + ln(S))` (read-only; valid after command 1) |

**Commands:**

| Cmd | Operation | States |
|-----|-----------|--------|
| `0x01` | Softmax tile (N=8): probabilities (Q0.8) + `z` | `SM_P1..SM_P4` |
| `0x02` | GELU | `GL_P1..GL_P3` |
| `0x04` | Streaming matmul | `MM_START`, `MM_WAIT` |
| `0x05` | SwiGLU silu group: `silu_s = sat12(round(x·σ >> 2))` into the 12-bit `scratch_a` | `SW_P1` |
| `0x06` | SwiGLU max group: max `|silu_s·up|` via the shared `piped_max` into `reg_sw_max` | `SW_P2`, `SW_P2_M` |
| `0x07` / `0x08` | RMSNorm sum-of-squares beat, first tile (clears the accumulator) / next tile (accumulates) | `RM_P1` / `RM_P1N` |
| `0x09` | RMSNorm scale (once per row): mean → ln → rounded −0.5× → sign-guarded exp → `reg_inv_rms` | `RM_P2`, `RM_P3` |
| `0x0A` | RMSNorm apply (per tile): `sat8(round(x·inv_rms >> 8))` | `RM_P4` |
| `0x0B` | SwiGLU row gain: `e` from `reg_sw_max` (consumes and clears it) | `SW_G` |
| `0x0C` | SwiGLU apply group: `sat8(round_shift(4·silu_s·up, 10−e))` | `SW_P4` |

**There is no command 3 any more** (the single-tile RMSNorm of earlier revisions). RMSNorm (rows of 8·tiles elements) and SwiGLU (rows of 8·groups elements) are **host-sequenced** in 8-lane groups: RMSNorm = `0x07`, `0x08`…, `0x09`, `0x0A` per tile; SwiGLU = pass 1 (`0x05`, `0x06` per group), `0x0B`, pass 2 (`0x05`, `0x0C` per group). See the header of `tr_soc_ctrl_int.sv` and `docs/llama/hardware-spec.md` §10; golden models are `rms_*` / `sw_*` in `emulation/tr_math_model.hpp`.

**Pass sequencing:** each command is a fixed sequence of states driving `tr_nonlinear_vpu`'s crossbar-select/enable/mode outputs pass-by-pass, gated on that pass's own `vpu_*_valid` handshake (the VPU's submodules have no external ready signal, so each pass waits out its producer's own valid pulse before advancing).

- **GL_P1..GL_P3** (GELU): `E = exp(-α|x|)` → `recip = 1/(1+E)` → `y = x·σ(x)`.
- **SM_P1..SM_P4** (Softmax): row max (shared `piped_max`) → `Σexp(x-max)` (via `mac_array_engine`) → `ln(Σ)` → `exp(x-max-ln(Σ))` (Q0.8 output). `z = sat8(max + ln Σ)` is the offset SM_P4 subtracts, exposed at `ADDR_SM_Z`.
- **RM_P1..RM_P4** (RMSNorm): `S += Σx²` over tiles → `mk = round(S >> (FRAC_W + log2N − 2))` → shared ln at BITS = FRAC_W+2 → rounded `−0.5·ln` → `InvRMS = exp(...)` (sign-guarded, see below) → wide `inv_rms` register (12 bits, 8 fractional bits) → `y = round(x·inv_rms >> 8)`, int8 saturate.
- **SW_P1, SW_P2, SW_G, SW_P4** (SwiGLU): σ ROM lookup and silu rounding → max `|silu·up|` → gain → apply with the per-row shift.

`scratch_a` (12 bits per lane; GELU uses the low 8) and `scratch_b` hold intermediate per-lane vectors between passes. `reg_scalar_max`/`reg_scalar_log` hold Softmax's row-max and log-sum-exp scalars, and RMSNorm's `−0.5·ln(mean)` scalar, between their own passes. `reg_inv_rms` (RMSNorm) and `reg_sw_max`/`reg_sw_e` (SwiGLU) are the new cross-command registers.

**RM_P3 sign-guard:** `tr_nonlinear_vpu`'s shared exp backbone only supports the decay/negative-domain case, but RMSNorm's log-domain argument can legitimately go positive for a small mean. RM_P3 forces it non-positive before the backbone (`rms_growth` flag), and the VPU turns the result into `inv_rms` through the `recip_rom` reciprocal table when the true value was positive (`1/E`, capped at 4095). The standalone `tr_rmsnorm.sv` baseline module (`rtl_baseline/tr_baseline/`) keeps the older single-tile algorithm.

### Parameters

| Parameter | Default | Description                                                    |
|-----------|---------|------------------------------------------------------------------|
| `N`       | `8`     | Vector dimension (lanes) of the driven VPU/datapath                |
| `W`       | `8`     | Data word width (`ctrl_scalar_sub_val`, `scratch_a`/`b`, `vpu_data_out`) |
| `ACC_W`   | `32`    | Accumulator width (`req_mult_out`, `vpu_dot_out`)                   |
| `INV_W`   | `12`    | RMSNorm `inv_rms` register width                                    |
| `SA_W`    | `12`    | SwiGLU silu intermediate width (`scratch_a`)                        |
| `WIDE_W`  | `20`    | Width of the VPU's wide vecmul / shared-max bus                     |
| `GELU_EN` | `0` (macro `GELU_ENABLE` → `1`) | 0 (default): command 2 (GELU) is ignored and the GELU-only controls (`write_scratch_b`, `sym_mode_en`, `bb_mode_pre_ln`, post-ln modes other than bypass / rounded −0.5×) are held constant so `scratch_b`, the symmetry modifier and the GELU datapath are optimized away |

---

## `tr_soc_ctrl_mx`

**File:** [rtl_soc/tr_soc_ctrl/tr_soc_ctrl_mx.sv](../../rtl_soc/tr_soc_ctrl/tr_soc_ctrl_mx.sv)

### Operation

**Not yet updated** for the commands, ports and registers added to `tr_soc_ctrl_int` for the Llama work (tiled RMSNorm, two-pass SwiGLU, `ADDR_SM_Z`, wide buses); it still follows the earlier command set. The MX counterpart of `tr_soc_ctrl_int` — same command dispatch and the same `GL_*`/`SM_*`/`RM_*` pass sequencing over `tr_nonlinear_vpu`'s crossbar, with two differences specific to the MX datapath:

1. **Double-buffered outputs.** Every FSM output goes through an explicit `nxt_*` "control pipeline register" stage instead of being driven straight from the combinational case block, adding one cycle of latency per state for MX's tighter timing closure.
2. **Exponent-based rescaling.** Drives `ctrl_enable_linear_shift` (the MX dynamic shifter's enable) instead of the INT variant's explicit `req_mult_out`/`req_shift_out` requantizer registers, since MX rescaling is exponent-based rather than an explicit multiply-shift. Softmax's SM_P1 max-reduction also waits a fixed 3-cycle latency instead of polling a `vpu_max_valid` handshake, since that signal isn't wired into this variant.

The RMSNorm sign-guard fix is the same as `tr_soc_ctrl_int`, applied here to the pipelined `nxt_*` outputs.

### Parameters

Same as `tr_soc_ctrl_int` (`N`, `W`, `ACC_W`).

---

## `tr_matmul_ctrl`

**File:** [rtl_soc/tr_matmul_ctrl.sv](../../rtl_soc/tr_matmul_ctrl.sv)

### Operation

Format-agnostic streaming-matmul sequencer (control only — no datapath instantiated here). Drives a shared linear datapath (`dot_product_engine` + requantizer) to compute `OUT[r] = requant(Σ_c A[r][c]·B[c])` over an `(num_row_tiles·M) × (num_ctiles·N)` matmul. Since this FSM only sequences the datapath rather than owning it, the same module is reused by both `tr_soc_top_int` and `tr_soc_top_mx` (each wires it to its own int/mx dot+requant pair).

Per output-row-tile: streams `num_ctiles` tiles (`clear_acc=1` on the first, `=0` to accumulate), then captures the result on the `num_ctiles`-th requantizer valid — the final accumulated dot. Using the valid *count* (not a fixed drain latency) keeps this independent of the datapath's pipeline depth.

### Parameters

| Parameter | Default | Description                                                       |
|-----------|---------|-----------------------------------------------------------------------|
| `M`       | `4`     | Output-tile height (rows per row-tile) of the driven `dot_product_engine` |
| `N`       | `8`     | Contraction-tile width (columns per c-tile) of the driven `dot_product_engine` |

### Ports (key control signals)

| Port            | Direction | Width | Description                              |
|-----------------|-----------|-------|---------------------------------------------|
| `clk`, `rst_n`  | input     | 1     | Clock / active-low reset                      |
| `start`         | input     | 1     | Kick off a streaming matmul                    |
| `num_row_tiles` | input     | 16    | Output rows ÷ `M`                             |
| `num_ctiles`    | input     | 16    | Contraction columns ÷ `N`                     |
| `busy`          | output    | 1     | Sequencer is running                           |
| `done`          | output    | 1     | Sequencer finished                             |
