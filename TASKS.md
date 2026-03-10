# Branch Tasks

## Changed / Added

### Renaming & Restructuring
*   **Folder**: `rtl/round-exp/` -> `rtl/tr_exp/` (reflects top-level module)
*   **File**: `rtl/round-exp/dcompose-round-exp.sv` -> `rtl/tr_exp/tr_exp.sv`
*   **Folder**: `tb/round-exp/` -> `tb/tr_exp/`
*   **File**: `tb/round-exp/round-exp_tb-full.sv` -> `tb/tr_exp/tr_exp_tb.sv`
*   **File**: `tb/round-exp/round-exp_tb.f` -> `tb/tr_exp/tr_exp.f`
*   **File**: `tb/online_sum/sum_x-max_tb.sv` -> `tb/online_sum/online_sum_tb.sv`
*   **File**: `tb/softmax/softmax.f` -> `tb/softmax/exp_sum.f`

### Fixes & Updates
*   `rtl/tr_exp/tr_exp.sv` + `rtl/tr_exp/round.sv`: Bug fixed.
*   `tb/tr_exp/tr_exp_tb.sv`: TB fixed.
*   `tb/tr_exp/tr_exp.f`: File names updated.
*   `rtl/vec_mac/vec_mac_su.sv`: Bug fixed.
*   `tb/vec_mac/vec_mac_tb.sv`: Compilation warnings fixed.
*   `tb/vec_mac/vec_mac_su_tb.sv`: New TB for `vec_mac_su.sv`.
*   `tb/softmax/exp_sum_tb.sv`: TB fixed.
*   `tb/softmax/exp_sum.f`: File names updated.
*   `rtl/tr_ln/tr_ln.sv`: File take from `rtl` branch.
*   `tb/tr_ln/tr_ln_tb.sv`: TB fixed.
*   `tb/tr_ln/tr_ln.f`: .f file created.
*   `rtl/tr_div/tr_div.sv`: File take from `rtl` branch.
*   `tb/tr_div/tr_div_tb.sv`: TB fixed.
*   `tb/tr_div/tr_div.f`: .f file created.
*   `tb/online_sum/piped_max_tb.sv`: TB extended.
*   `tb/online_sum/online_sum_tb.sv`: TB fixed.
*   `tb/online_sum/online_sum.f`: .f file created.


## Pending Tasks

### Cleanup: `rtl/tr_exp`
- [ ] Check/remove `exp_wapper.sv`
- [ ] Check/remove `tr_exp_vec.sv`

### Cleanup: `tb/tr_exp`
- [ ] Check/remove `round-exp_tb.sv`

### Cleanup: `tb/tr-exp-tb`
- [ ] Check/remove `tr-exp-tb.sv`

### Cleanup: `tb/`
- [ ] Check/remove `tr_exp_tb.sv`

### Cleanup: `rtl/quantization/`
- [ ] Check/remove duplicate `requant_timing.sv`/`requant_unit.sv`

### Inquire: Naming
- [ ] `online_sum` folder+file name
- [ ] `qlinear` folder+file name
- [ ] `quantization` folder name
- [ ] `tr_div` file name
