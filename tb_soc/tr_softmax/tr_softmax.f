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
../rtl_soc/tr_nonlinear_vpu/peripheral_modules/piped_max.sv
../rtl_soc/tr_nonlinear_vpu/peripheral_modules/scalar_sub.sv
../rtl_soc/tr_nonlinear_vpu/peripheral_modules/vec_mul.sv

../rtl_soc/tr_softmax/tr_softmax.sv

//tb
../tb_soc/tr_softmax/tr_softmax_tb.sv