// global config
-f ../scripts/xrun_config.f
// exponent decomposition
../rtl/tr_exp/tr_exp.sv 
../rtl/tr_exp/round.sv 
../rtl/tr_exp/quadratic_divider.sv
// tr reciprocal
../rtl/tr_reciprocal/tr_reciprocal.sv 
// tr ln
../rtl/tr_ln/tr_ln.sv 
// vector mac
../rtl/vec_mac/vec_mac_su.sv
// online sum
../rtl/online_sum/online_sum.sv
../rtl/online_sum/piped_max.sv
// tr_gelu
../rtl/tr_gelu/tr_gelu.sv
// tb
../tb/tr_gelu/tr_gelu_tb.sv