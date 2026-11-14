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
