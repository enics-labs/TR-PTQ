// global config
-f ../scripts/xrun_config.f
// exponent decomposition
../rtl/tr_exp/tr_exp.sv 
../rtl/tr_exp/round.sv 
../rtl/tr_exp/quadratic_divider.sv
// tr div
../rtl/tr_div/tr_div.sv 
// tr ln
../rtl/tr_ln/tr_ln.sv 
// exp sum
// ../rtl/softmax/exp_sum.sv
// vector mac
../rtl/vec_mac/vec_mac_su.sv
// online sum
../rtl/online_sum/online_sum.sv
../rtl/online_sum/piped_max.sv
// softmax
../rtl/softmax/softmax_engine.sv
// tb
../tb/softmax/softmax_engine_tb.sv