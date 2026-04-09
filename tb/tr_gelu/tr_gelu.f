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

// tr_gelu
../rtl/tr_gelu/tr_gelu_trick.sv
// tb
../tb/tr_gelu/tr_gelu_tb.sv