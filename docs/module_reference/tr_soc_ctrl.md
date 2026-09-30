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
| `0x00`  | `ADDR_CMD`        | Write triggers a command: 1=Softmax, 2=GELU, 3=RMSNorm, 4=streaming matmul |
| `0x04`  | `ADDR_STATUS`     | `{done, busy}`                                                 |
| `0x08`  | `ADDR_REQ_MULT`   | Requantizer multiplier "M"                                     |
| `0x0C`  | `ADDR_REQ_SHIFT`  | Requantizer shift "S"                                          |
| `0x10`  | `ADDR_MM_ROWS`    | Streaming-matmul row-tile count                                 |
| `0x14`  | `ADDR_MM_CTILES`  | Streaming-matmul contraction-tile count                         |

**Pass sequencing:** each command is a fixed sequence of states (`GL_*`, `SM_*`, `RM_*`) driving `tr_nonlinear_vpu`'s crossbar-select/enable/mode outputs pass-by-pass, gated on that pass's own `vpu_*_valid` handshake (the VPU's submodules have no external ready signal, so each pass waits out its producer's own valid pulse before advancing).

- **GL_P1..GL_P3** (GELU): `E = exp(-α|x|)` → `recip = 1/(1+E)` → `y = x·σ(x)`.
- **SM_P1..SM_P4** (Softmax): row max → `Σexp(x-max)` (via `mac_array_engine`) → `ln(Σ)` → `exp(x-max-ln(Σ))` (Q0.8 output).
- **RM_P1..RM_P4** (RMSNorm): `S = Σx²` → `-0.5·ln(S)` → `InvRMS = exp(...)` (sign-guarded, see below) → `y = x·InvRMS`.

`scratch_a`/`scratch_b` hold intermediate per-lane vectors between passes (e.g. GELU's `exp(α(x))` and reciprocal-sigmoid results). `reg_scalar_max`/`reg_scalar_log` hold Softmax's row-max and log-sum-exp scalars, and RMSNorm's `log(Σx²)` scalar, between their own passes.

**RM_P3 sign-guard:** `tr_nonlinear_vpu`'s shared exp backbone only supports the decay/negative-domain case, but RMSNorm's log-domain argument can legitimately go positive for a small `Σx²`. RM_P3 forces it non-positive before the backbone, then corrects the result back via `recip_lut` when the true value was positive — see the inline comment at `rm_ctrl_scalar_raw` in the RTL for the full derivation. This same fix is mirrored in the standalone `tr_rmsnorm.sv` baseline module (`rtl_baseline/tr_baseline/`).

### Parameters

| Parameter | Default | Description                                                    |
|-----------|---------|------------------------------------------------------------------|
| `N`       | `8`     | Vector dimension (lanes) of the driven VPU/datapath                |
| `W`       | `8`     | Data word width (`ctrl_scalar_sub_val`, `scratch_a`/`b`, `vpu_data_out`) |
| `ACC_W`   | `32`    | Accumulator width (`req_mult_out`, `vpu_dot_out`)                   |

---

## `tr_soc_ctrl_mx`

**File:** [rtl_soc/tr_soc_ctrl/tr_soc_ctrl_mx.sv](../../rtl_soc/tr_soc_ctrl/tr_soc_ctrl_mx.sv)

### Operation

The MX counterpart of `tr_soc_ctrl_int` — same command dispatch and the same `GL_*`/`SM_*`/`RM_*` pass sequencing over `tr_nonlinear_vpu`'s crossbar, with two differences specific to the MX datapath:

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
