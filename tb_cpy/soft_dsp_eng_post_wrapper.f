// Generic SystemVerilog file list for simulating soft_dsp_eng_post_wrapper_tb.
// Usage examples:
//   iverilog -g2012 -o simv -f tb/soft_dsp_eng_post_wrapper_tb.f
//   vcs -sverilog -f tb/soft_dsp_eng_post_wrapper_tb.f
//   verilator --sv --binary -f tb/soft_dsp_eng_post_wrapper_tb.f

// global config
-f ../scripts/xrun_config.f

../rtl_cpy/tr_exp/tr_exp_wrapper.sv
../rtl_cpy/tr_exp/tr_exp.sv
../rtl_cpy/tr_exp/q4_4_round_neg.sv
../rtl_cpy/tr_exp/q4_4_quadratic_divider.sv

../rtl_cpy/online_sum/piped_max.sv
../rtl_cpy/online_sum/max_sub.sv

../rtl_cpy/vec_mul/vec_mul.sv

../rtl_cpy/tr_ln/tr_ln.sv

../rtl_cpy/divu_int/divu_int.sv

../rtl_cpy/soft_dsp_eng_post_wrapper.sv

../tb_cpy/soft_dsp_eng_post_wrapper_tb.sv