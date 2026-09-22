#!/bin/tcsh
# Run from workspace/: tcsh ../tb_soc/tr_exp_alu/sweep_exp_alu.sh
# Sweeps tr_exp_alu (round.sv + quadratic_divider.sv) across all 6 target
# formats, comparing RTL output to the Python golden model exhaustively.
setenv CDS_LIC_FILE 5280@enicsw01
setenv DISPLAY ""

set TBDIR = ../tb_soc/tr_exp_alu
set TB = $TBDIR/tr_exp_alu_tb.sv

foreach fmt (8:4:Q4.4 10:6:Q4.6 12:8:Q4.8 10:4:Q6.4 12:4:Q8.4 16:4:Q12.4)
    set parts = `echo $fmt | tr ':' ' '`
    set W = $parts[1]
    set F = $parts[2]
    set NAME = $parts[3]

    echo "=== $NAME (WIDTH=$W FRAC_W=$F) ==="

    sed -i "s/localparam int WIDTH     = [0-9]*;/localparam int WIDTH     = $W;/" $TB
    sed -i "s/localparam int FRAC_W    = [0-9]*;/localparam int FRAC_W    = $F;/" $TB

    python3 $TBDIR/gen_exp_alu_golden.py $W $F

    rm -rf xcelium.d worklib
    xrun -clean -f $TBDIR/tr_exp_alu.f > xrun_${NAME}.log
    if ($status != 0) then
        echo "  [FATAL] xrun failed, see xrun_${NAME}.log"
        continue
    endif

    python3 $TBDIR/compare_exp_alu.py
end
