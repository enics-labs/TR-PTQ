#!/bin/tcsh
# Run from workspace/: tcsh ../tb_soc/tr_nonlinear_vpu/sweep_tr_nonlinear_vpu.sh
# Sweeps tr_nonlinear_vpu's full GELU/Softmax/RMSNorm(decay) sequences
# across all 6 target formats, comparing RTL output to the Python golden
# model (precision_analysis_v3.py, native I/O mode).
setenv CDS_LIC_FILE 5280@enicsw01
setenv DISPLAY ""

set TBDIR = ../tb_soc/tr_nonlinear_vpu
set TB = $TBDIR/tr_nonlinear_vpu_tb.sv

# name:W_VEC:FRAC_W:int_bits:RM_CONST_LN_SQRT_N (round(0.5*ln(8)*2^FRAC_W))
foreach fmt (Q4.4:8:4:4:17 Q4.6:10:6:4:67 Q4.8:12:8:4:266 Q6.4:10:4:6:17 Q8.4:12:4:8:17 Q12.4:16:4:12:17)
    set parts = `echo $fmt | tr ':' ' '`
    set NAME = $parts[1]
    set W = $parts[2]
    set F = $parts[3]
    set IB = $parts[4]
    set C = $parts[5]

    echo "=== $NAME (W_VEC=$W FRAC_W=$F RM_CONST=$C) ==="

    sed -i "s/localparam int W_VEC             = [0-9]*;/localparam int W_VEC             = $W;/" $TB
    sed -i "s/localparam int FRAC_W            = [0-9]*;/localparam int FRAC_W            = $F;/" $TB
    sed -i "s/localparam int RM_CONST_LN_SQRT_N = [0-9]*;/localparam int RM_CONST_LN_SQRT_N = $C;/" $TB

    python3 $TBDIR/gen_vpu_golden.py $IB $F

    rm -rf xcelium.d worklib
    xrun -clean -f $TBDIR/tr_nonlinear_vpu.f > xrun_vpu_${NAME}.log
    if ($status != 0) then
        echo "  [FATAL] xrun failed, see xrun_vpu_${NAME}.log"
        continue
    endif

    python3 $TBDIR/compare_vpu.py
end
