#!/bin/tcsh
# Run from workspace/: tcsh ../tb_soc/peripheral_modules/alpha_stabilizer/sweep_alpha_stabilizer.sh
# Sweeps alpha_stabilizer_tb's self-contained exhaustive check across all 6
# target formats (no external golden files -- the testbench self-checks
# against its own inline golden_model()).
setenv CDS_LIC_FILE 5280@enicsw01
setenv DISPLAY ""

set TBDIR = ../tb_soc/peripheral_modules/alpha_stabilizer
set TB = $TBDIR/alpha_stabilizer_tb.sv

foreach fmt (8:4:Q4.4 10:6:Q4.6 12:8:Q4.8 10:4:Q6.4 12:4:Q8.4 16:4:Q12.4)
    set parts = `echo $fmt | tr ':' ' '`
    set W = $parts[1]
    set F = $parts[2]
    set NAME = $parts[3]

    echo "=== $NAME (W=$W FRAC_W=$F) ==="

    sed -i "s/localparam int W      = [0-9]*;/localparam int W      = $W;/" $TB
    sed -i "s/localparam int FRAC_W = [0-9]*;/localparam int FRAC_W = $F;/" $TB

    rm -rf xcelium.d worklib
    xrun -clean -f $TBDIR/alpha_stabilizer.f > xrun_alpha_${NAME}.log
    grep -q "\[SUCCESS\]" xrun_alpha_${NAME}.log
    if ($status == 0) then
        echo "  [PASS]"
    else
        echo "  [FAIL] see xrun_alpha_${NAME}.log"
    endif
end
