# Standalone operator modules — `rtl_baseline`

Dedicated, self-contained blocks of the nonlinear operators, used as area and energy references for the shared `tr_nonlinear_vpu`. Per operator, two modules with the same command interface, so one testbench style and one golden model drive them:

| Directory | Style |
|---|---|
| `rtl_baseline/tr_baseline/` | the arithmetic of the production flow (`rtl_soc/`) built as a dedicated block, without the shared backbone and crossbar |
| `rtl_baseline/ibert_baseline/` | integer-only, divider-based (I-BERT: Newton integer square root, integer divisions), at the same output precision |

Testbenches: `tb_baseline/<style>/<module>/<module>_tb.sv` (correctness, bit-exact against the C++ golden model of `emulation/`) and `<module>_energy_tb.sv` (energy pilot).

---

## RMSNorm: `tr_rmsnorm`, `ibert_rmsnorm`

Both share the sum-of-squares front end [rmsnorm_sumsq.sv](../../rtl_baseline/rmsnorm_sumsq.sv) (4 pipeline stages: input registers, 8 squares, adder tree, accumulate), so the comparison isolates how 1/rms is formed and applied.

### Command interface

Host-sequenced like the controller of the production flow:

| `cmd` | Name | Action |
|---|---|---|
| 0 | ACC_FIRST | S = sum of x² over the 8-lane tile |
| 1 | ACC | S += sum of x² over the tile |
| 2 | SCALE | scale register from S (`log2n` selects N, stable from ACC to SCALE) |
| 3 | APPLY | y = round(x · scale) per lane, 12-bit signed Q5.7 |

`out_valid` pulses when a command has completed; `scale_o` is the scale register (`inv` for `tr_rmsnorm`, `R` for `ibert_rmsnorm`); `y_out` is valid after APPLY.

### `tr_rmsnorm`

[rtl_baseline/tr_baseline/tr_rmsnorm.sv](../../rtl_baseline/tr_baseline/tr_rmsnorm.sv). SCALE: mean `mk = round(S >> (log2n + 2))` saturated to 16 bits, leading-one finder and normalizer (the pair `tr_ln_alu` provides to `rsq_unit` in the VPU), [rsq_unit](peripheral_modules.md) (2 × 32 entries × 12 bits, one shift) → `inv`. APPLY: 8 multipliers `x · inv`, rounded shift by 5, saturation to 12 bits. Golden: `rms_acc_tile`, `rms2_scale`, `rms2_apply_tile` (C++ mode `rmsnorm2`).

### `ibert_rmsnorm`

[rtl_baseline/ibert_baseline/ibert_rmsnorm.sv](../../rtl_baseline/ibert_baseline/ibert_rmsnorm.sv), with the restoring divider [ibert_divider.sv](../../rtl_baseline/ibert_baseline/ibert_divider.sv) (WIDTH cycles per division). SCALE: `msq = round(S · 4^F / N)` with F = 5, `R = isqrt(msq)` by Newton iteration (I-BERT Algorithm 4, `x ← (x + msq / x) >> 1` through the divider until x stops decreasing); R is the rms code × 2^F. APPLY: one division per lane through the same divider, `y = sign · min(round(|x| · 2^(7+F) / R), 2047 | 2048)`, lanes sequential. All-zero input gives R = 0 and zero output. Golden: `ibert_rmsnorm_scale`, `ibert_rmsnorm_apply_tile` (C++ mode `ibert_rmsnorm`). Accuracy against exact RMSNorm on Llama-like rows: mean error 0.10 LSB of the Q5.7 output (0.59 LSB for `tr_rmsnorm`).

### Verification

```
cd workspace
python ../verification/verify_block.py tr_rmsnorm       # 65,696 / 65,696 exact
python ../verification/verify_block.py ibert_rmsnorm    # 65,696 / 65,696 exact (long: sequential divisions, about 8,200 tiles)
```

The 160 rows (N = 8 … 2048) are those of the production RMSNorm regression. Run the long one through the cluster wrapper (`qrun`).

### Energy pilots

`<module>_energy_tb.sv`: the representative 8-element vector of the other pilots, ACC_FIRST + SCALE + APPLY back to back, the cycle count from the first command to the last result, and a VCD of exactly that window (`<module>_power_activity.vcd` in `workspace/`). Energy = Total_Power × N_cycles × T_clock ([verification.md](../verification.md)).

| Pilot, one 8-element tile | Cycles |
|---|---|
| `tr_rmsnorm` | 20 (including one issue cycle and one done cycle per command) |
| `ibert_rmsnorm` | 369 (Newton iterations plus 8 sequential lane divisions, 25 cycles each) |

The cycle definition (command issue and completion included) differs from the shared-VPU pilot, which counts the passes of its own sequencing.

### Synthesis

Top module = module name, clock 3.3 ns. File lists: `tr_rmsnorm`: `rtl_baseline/rmsnorm_sumsq.sv`, `rtl_soc/tr_nonlinear_vpu/peripheral_modules/rsq_unit.sv`, `rtl_baseline/tr_baseline/tr_rmsnorm.sv`. `ibert_rmsnorm`: `rtl_baseline/rmsnorm_sumsq.sv`, `rtl_baseline/ibert_baseline/ibert_divider.sv`, `rtl_baseline/ibert_baseline/ibert_rmsnorm.sv` (the `.f` files in the testbench directories list the same).

---

## SwiGLU: `tr_swiglu`, `ibert_swiglu`

Both share the back end [swiglu_core.sv](../../rtl_baseline/swiglu_core.sv) (silu register, 8 products `silu · up`, running max of `|silu · up|`, row gain, apply), so they differ only in how the 12-bit silu of a gate group is produced.

### Command interface

Host-sequenced in 8-lane groups, like the controller of the production flow (pass 1 = SILU + MAX per group, GAIN once, pass 2 = SILU + APPLY per group):

| `cmd` | Name | Action |
|---|---|---|
| 0 | SILU | gate group `x_in` → silu register: `silu = sat12((x · σ(x) + 2) >> 2)`, σ in Q.6 |
| 1 | MAX | up group `x_in`: running max of `|silu · up|` |
| 2 | GAIN | row gain `e` = largest k in [-8, 10] with `(max << 2) << k <= 2047 << 10`; clears the max |
| 3 | APPLY | up group `x_in`: `y = sat12(round_shift((silu · up) << 2, 10 - e))`, 12-bit signed |

`out_valid` pulses when a command has completed; `scale_o` is the row gain `e` (signed); `y_out` is valid after APPLY.

### `tr_swiglu`

[rtl_baseline/tr_baseline/tr_swiglu.sv](../../rtl_baseline/tr_baseline/tr_swiglu.sv). σ from the 64-entry symmetric table ([sigma_rom](peripheral_modules.md): index `|x| >> 1`, negative codes read `64 - entry`); three pipeline stages (register, table lookup, multiply–round–saturate). Golden: `sw2_silu`, `sw_abs_prod`, `sw2_gain`, `sw2_apply` (C++ mode `swiglu2`).

### `ibert_swiglu`

[rtl_baseline/ibert_baseline/ibert_swiglu.sv](../../rtl_baseline/ibert_baseline/ibert_swiglu.sv), with the restoring divider [ibert_divider.sv](../../rtl_baseline/ibert_baseline/ibert_divider.sv). σ from the I-BERT integer exponential (Algorithm 3) and one division per lane: with `a = 2|x|` (one extra input fractional bit, scale 1/32), `z = floor(a / 22)`, `p = 22 z − a`, `e = ((p + 43)² + 983) >> z`, `σ = N / D` with `D = 2859 + e` and `N = 2859` for `x ≥ 0` or `N = e` for `x < 0`, computed as `(64 N + D/2) / D`. The division by the constant 22 and the polynomial are combinational; the 18-bit sigma division is iterative and shared by the 8 lanes (sequential). Golden: `ibert_sigma6`, `ibert_silu` (then `sw2_gain`, `sw2_apply`; C++ mode `ibert_swiglu`).

Accuracy of σ against exact `64 / (1 + e^-x)` over all 256 gate codes: mean error 0.217 LSB of Q.6 and maximum 0.685 for `ibert_swiglu`; 0.256 and 1.000 for the table. With the plain Q4.4 input scale the polynomial gives 0.329 / 1.163, which is why one extra fractional bit is used. The divider variant is therefore at least as accurate as the table.

### Verification

```
cd workspace
python ../verification/verify_block.py tr_swiglu       # bit-exact against swiglu2
python ../verification/verify_block.py ibert_swiglu    # bit-exact against ibert_swiglu
```

The 40 rows (n = 8 … 8192; heavy-tailed gate / up, gate outliers at the int8 limits, an all-zero up row at the gain cap, tiny and saturating products) are those of the production SwiGLU regression. Run the long ones through the cluster wrapper (`qrun`).

### Energy pilots

`<module>_energy_tb.sv`: one representative 8-lane group (gate vector of the other pilots, up vector of the shared-VPU SwiGLU pilot), the five commands of the two-pass flow back to back (SILU, MAX, GAIN, SILU, APPLY), the cycle count from the first command to the last result, and a VCD of that window (`<module>_power_activity.vcd` in `workspace/`).

| Pilot, one 8-lane group, two passes | Cycles |
|---|---|
| `tr_swiglu` | 30 (command issue and completion included) |
| `ibert_swiglu` | 410 (8 sequential sigma divisions of 18 cycles in each of the two SILU commands) |

### Synthesis

Top module = module name, clock 3.3 ns. File lists: `tr_swiglu`: `rtl_baseline/swiglu_core.sv`, `rtl_soc/tr_nonlinear_vpu/peripheral_modules/sigma_rom.sv`, `rtl_baseline/tr_baseline/tr_swiglu.sv`. `ibert_swiglu`: `rtl_baseline/swiglu_core.sv`, `rtl_baseline/ibert_baseline/ibert_divider.sv`, `rtl_baseline/ibert_baseline/ibert_swiglu.sv`. Since `genus.tcl` reads every `.sv` under `rtl_soc/` and `rtl_baseline/`, only the top module name is needed.
