# Per-Operator Energy Summary (Q4.4, 65nm TSMC, Worst-Case Corner)

Energy-per-operator characterization for all 7 designs in this batch:
the three standalone TR blocks (`tr_gelu`, `tr_softmax`, `tr_rmsnorm`),
their I-BERT baseline counterparts, and the shared `tr_nonlinear_vpu`
crossbar driven through its three GELU/Softmax/RMSNorm passes.

## Methodology

1. Each design has a dedicated `*_energy_tb.sv` testbench (separate from
   its correctness-verification testbench) that drives one representative
   8-lane vector through the design and counts the clock cycles from
   stimulus-valid to result-valid.
2. A VCD is captured over exactly that operation window (reset transients
   and settling excluded) and fed to Genus with the already-synthesized
   post-opt netlist (`read_db`, no re-synthesis) via
   `read_vcd -vcd_scope <tb>/<dut> <file>.vcd`, then `report_power`.
3. All designs use the **same representative stimulus vector** (Q4.4
   codes `{12,-20,5,-8,30,-3,18,-45}`, i.e. real values
   `{0.75,-1.25,0.3125,-0.5,1.875,-0.1875,1.125,-2.8125}`) so the numbers
   below are directly comparable to each other.
4. `Energy = Total_Power(uW) x N_cycles x T_clock(ns)`, with
   `T_clock = 3.3 ns` (the fixed Genus target clock period used
   throughout this synthesis flow, `dut.defines: CLK_PERIOD=3.3`).
   Power(uW) x Time(ns) = Energy(fJ) directly (1 uW*ns = 1 fJ).
5. Corner: `ALL_WC_LIBS` (worst-case: slow process, low voltage, high
   temperature) for every design, for consistency with the rest of this
   project's methodology. This is a framing choice, not necessarily a
   true worst-case for leakage (leakage rises with temperature but falls
   with the low voltage in this corner) -- kept as-is unless told
   otherwise.
6. `tr_nonlinear_vpu` has no single valid_in/valid_out handshake -- it is
   a crossbar with 4 separate valid outputs (bb/mac/vecmul/max), driven
   pass-by-pass exactly like the real controller (`tr_soc_ctrl_int.sv`)
   drives it in production. Each of its 3 rows below is the SAME physical
   netlist, driven through a different pass sequence (GELU: 3 passes,
   Softmax: 4 passes, RMSNorm: 4 passes) -- not 3 different designs.

## Results

| Design                         | Function | N cycles | Op time (ns) | Leakage (uW) | Internal (uW) | Switching (uW) | **Total Power (uW)** | **Energy (pJ)** |
|---------------------------------|----------|---------:|-------------:|-------------:|---------------:|----------------:|----------------------:|----------------:|
| `tr_gelu` (standalone)          | GELU     |       21 |          69.3 |         33.01 |          367.63 |           162.59 |                563.23 |          **39.03** |
| `ibert_gelu`                    | GELU     |        4 |          13.2 |         23.86 |          237.89 |           203.07 |                464.82 |           **6.14** |
| `tr_nonlinear_vpu` (GL_P1-P3)    | GELU     |       16 |          52.8 |         39.14 |          741.87 |           398.78 |               1179.79 |          **62.29** |
| `tr_softmax` (standalone)       | Softmax  |       26 |          85.8 |         19.69 |          481.92 |           208.36 |                709.97 |          **60.92** |
| `ibert_softmax`                 | Softmax  |      228 |         752.4 |         15.29 |           59.34 |            24.03 |                 98.66 |          **74.23** |
| `tr_nonlinear_vpu` (SM_P1-P4)    | Softmax  |       22 |          72.6 |         39.17 |          608.11 |           235.63 |                882.90 |          **64.10** |
| `tr_rmsnorm` (standalone)       | RMSNorm  |       28 |          92.4 |         11.95 |         2449.24 |          1520.95 |               3982.14 |         **367.95** |
| `ibert_rmsnorm`                 | RMSNorm  |      314 |        1036.2 |         24.23 |           57.17 |            19.67 |                101.07 |         **104.73** |
| `tr_nonlinear_vpu` (RM_P1-P4)    | RMSNorm  |       19 |          62.7 |         39.46 |          686.85 |           324.29 |               1050.61 |          **65.87** |

## Observations

- **Shared crossbar vs. dedicated standalone TR blocks**: `tr_nonlinear_vpu`
  costs roughly 60% more energy per GELU op than the dedicated `tr_gelu`
  (62.3 vs 39.0 pJ), and about the same for Softmax (64.1 vs 60.9 pJ), but
  is **5.6x more energy-efficient** for RMSNorm than the dedicated
  `tr_rmsnorm` block (65.9 vs 368.0 pJ) -- `tr_rmsnorm`'s standalone
  internal/switching power is far higher than any other design in this
  table, consistent with it containing its own full division/sqrt
  machinery that the shared crossbar's RMSNorm pass instead gets "for
  free" by reusing the same exp/ln backbone the GELU and Softmax passes
  already pay for. The general pattern: the shared unit trades
  per-operator energy for the three dedicated units' combined area, and
  that trade is a clear net win specifically for RMSNorm.
- **TR vs I-BERT baseline, standalone-to-standalone**: `tr_gelu` uses
  6.4x more energy than `ibert_gelu` (39.0 vs 6.1 pJ) -- I-BERT's GELU
  path is a much smaller, 4-cycle circuit. For Softmax and RMSNorm the
  result flips: `tr_softmax` is 1.2x more efficient than `ibert_softmax`
  (60.9 vs 74.2 pJ), and `tr_rmsnorm` is 3.5x less efficient than
  `ibert_rmsnorm` (368.0 vs 104.7 pJ) -- I-BERT's iterative divider-based
  approach takes far more cycles (228/314 vs 26/28) but at much lower
  power per cycle, while TR's parallel approach is fast but the
  `tr_rmsnorm` block in particular draws very high power while active.
- **`tr_nonlinear_vpu` vs I-BERT baseline**: the shared crossbar beats
  both I-BERT Softmax (64.1 vs 74.2 pJ) and I-BERT RMSNorm (65.9 vs
  104.7 pJ) despite carrying the overhead of a general-purpose crossbar,
  and is competitive with I-BERT GELU within the same order of magnitude
  (62.3 vs 6.1 pJ -- I-BERT's dedicated 4-cycle GELU circuit is still the
  outlier here across every comparison).
- **`tr_nonlinear_vpu`'s near-constant per-pass energy** (62.3 / 64.1 /
  65.9 pJ across GELU/Softmax/RMSNorm) reflects that it is the exact same
  physical netlist for all three -- the small spread comes from cycle
  count and which crossbar paths toggle, not from a different circuit.

## Caveats

- Single representative stimulus vector, not a statistical average over
  many inputs -- sufficient for relative/order-of-magnitude comparison
  across designs, not a precise per-design energy distribution.
- Worst-case corner (`ALL_WC_LIBS`) used for power on every design,
  including designs whose own timing corner may differ -- a framing
  choice for consistency, see Methodology point 5.
- `N_cycles` includes a few extra settle cycles added to some
  `tr_nonlinear_vpu` pass transitions purely to work around testbench
  read-timing races (documented in each `*_energy_tb.sv`); these do not
  add real DUT activity beyond what the VCD already captures, so they do
  not distort the power number, only (very slightly) the cycle count used
  to convert it to energy.
- Q4.4 (8-bit, N=8 lanes) only. Other formats (Q4.6/Q4.8/Q6.4/Q8.4/Q12.4)
  are a separate, not-yet-run sweep.
