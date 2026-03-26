// global config
-f ../scripts/xrun_config.f
// exponent decomposition
../rtl/tr_exp/tr_exp.sv 
../rtl/tr_exp/round.sv 
../rtl/tr_exp/quadratic_divider.sv
// tr div
../rtl/tr_reciprocal/tr_reciprocal.sv 
../rtl/tr_reciprocal/classic_reciprocal.sv 
../rtl/tr_reciprocal/divu_int.sv 
// tr ln
../rtl/tr_ln/tr_ln.sv 
// vector mac
../rtl/vec_mac/vec_mac_su.sv
// online sum
../rtl/online_sum/online_sum.sv
../rtl/online_sum/piped_max.sv
../rtl/online_sum/max_sub.sv
// tr softmax
../rtl/tr_softmax/tr_softmax.sv
// tb
../tb/tr_softmax/tr_softmax_tb.sv