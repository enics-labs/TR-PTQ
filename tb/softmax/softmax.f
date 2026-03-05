// global config
-f ../scripts/xrun_config.f
// exponent decomposition
../rtl/round-exp/dcompose-round-exp.sv 
../rtl/round-exp/round.sv 
../rtl/round-exp/quadratic_divider.sv
// exp sum
../rtl/softmax/exp_sum.sv
// vector mac
../rtl/vec_mac/vec_mac_su.sv
// tb
../tb/softmax/exp_sum_tb.sv