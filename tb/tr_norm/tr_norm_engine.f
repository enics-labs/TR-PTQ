// global config
-f ../scripts/xrun_config.f
// exponent decomposition
../rtl/tr_exp/tr_exp.sv 
../rtl/tr_exp/round.sv 
../rtl/tr_exp/quadratic_divider.sv
// tr ln
../rtl/tr_ln/tr_ln.sv 
// tr reciprocal
../rtl/tr_reciprocal/tr_reciprocal.sv
// tr_norm
../rtl/tr_norm/tr_norm.sv
// tr_norm_engine
../rtl/tr_norm/tr_norm_engine.sv
// tb
../tb/tr_norm/tr_norm_engine_tb.sv