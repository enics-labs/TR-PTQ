// global config
-f ../scripts/xrun_config.f

// tr_backbone
../rtl_soc/tr_nonlinear_vpu/tr_backbone/pre_ln_modifier.sv
../rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_ln_alu.sv
../rtl_soc/tr_nonlinear_vpu/tr_backbone/post_ln_modifier.sv
../rtl_soc/tr_nonlinear_vpu/tr_backbone/tr_exp_alu.sv
../rtl_soc/tr_nonlinear_vpu/tr_backbone/round.sv
../rtl_soc/tr_nonlinear_vpu/tr_backbone/quadratic_divider.sv

// peripheral_modules
../rtl_soc/tr_nonlinear_vpu/peripheral_modules/shared_lut_rom.sv
../rtl_soc/tr_nonlinear_vpu/peripheral_modules/alpha_stabilizer.sv
../rtl_soc/tr_nonlinear_vpu/peripheral_modules/symmetry_modifier.sv
../rtl_baseline/tr_baseline/vec_mul.sv

../rtl_baseline/tr_baseline/tr_gelu.sv

// tb
../tb_baseline/tr_baseline/tr_gelu/tr_gelu_energy_tb.sv
