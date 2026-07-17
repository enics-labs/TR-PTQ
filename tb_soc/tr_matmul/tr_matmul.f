// global config
-f ../scripts/xrun_config.f

../rtl_soc/mult_engines/mac_array_engine.sv
../rtl_soc/dot_product_engine/dot_product_engine.sv
../rtl_soc/requantize_engine/requantize_engine_int.sv

// tb
../tb_soc/tr_matmul/tr_matmul_tb.sv