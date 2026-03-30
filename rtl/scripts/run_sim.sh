irun -gui -debug \
    -input ../scripts/restore.tcl \
    -nospecify \
    -timescale 1ns/1ps \
    ../sourcecode/tb/<your_DUT_module_name>_tb.v \
    ../export/post_synth/<your_DUT_module_name>.v
