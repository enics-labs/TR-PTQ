// global config
-f ../scripts/xrun_config.f
// online_sum
../rtl/online_sum/max_sub.sv 
../rtl/online_sum/online_sum.sv 
../rtl/online_sum/piped_max.sv
// exponent decomposition
../rtl/tr_exp/tr_exp.sv 
../rtl/tr_exp/round.sv 
../rtl/tr_exp/quadratic_divider.sv
// tb
../tb/online_sum/online_sum_tb.sv