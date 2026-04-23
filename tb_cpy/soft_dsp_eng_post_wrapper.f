// Usage example:
// cd tr-vit/workspace
// xrun -f ../tb_cpy/soft_dsp_eng_post_wrapper_tb.f

// global config
-f ../scripts/xrun_config.f

../rtl_cpy/tr_exp/tr_exp_wrapper.sv
../rtl_cpy/tr_exp/tr_exp.sv
../rtl_cpy/tr_exp/q4_4_round_neg.sv
../rtl_cpy/tr_exp/q4_4_quadratic_divider.sv

../rtl_cpy/piped_max/piped_max.sv
../rtl_cpy/max_sub/max_sub.sv

../rtl_cpy/vec_mul/vec_mul.sv

../rtl_cpy/tr_ln/tr_ln.sv

../rtl_cpy/divu_int/divu_int.sv

../rtl_cpy/soft_dsp_eng_post_wrapper.sv

../tb_cpy/soft_dsp_eng_post_wrapper_tb.sv