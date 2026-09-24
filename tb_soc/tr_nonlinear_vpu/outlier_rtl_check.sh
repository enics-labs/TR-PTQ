#!/bin/tcsh
# Run from workspace/: tcsh ../tb_soc/tr_nonlinear_vpu/outlier_rtl_check.sh
# RTL vs model on the outlier-grid stimulus (all six formats x five magnitudes).
setenv CDS_LIC_FILE 5280@enicsw01
setenv DISPLAY ""
set TBDIR = ../tb_soc/tr_nonlinear_vpu
set TB = $TBDIR/tr_nonlinear_vpu_tb.sv
foreach fmt (Q4.4:8:4:4:17:16 Q4.6:10:6:4:68:16 Q4.8:12:8:4:272:18 Q6.4:10:4:6:17:16 Q8.4:12:4:8:17:16 Q12.4:16:4:12:17:16)
    set parts = `echo $fmt | tr ':' ' '`
    set NAME = $parts[1]
    set W = $parts[2]
    set F = $parts[3]
    set IB = $parts[4]
    set C = $parts[5]
    set WM = $parts[6]
    sed -i "s/localparam int W_VEC             = [0-9]*;/localparam int W_VEC             = $W;/" $TB
    sed -i "s/localparam int FRAC_W            = [0-9]*;/localparam int FRAC_W            = $F;/" $TB
    sed -i "s/localparam int RM_CONST_LN_SQRT_N = [0-9]*;/localparam int RM_CONST_LN_SQRT_N = $C;/" $TB
    sed -i "s/localparam int W_MAC             = [0-9]*;/localparam int W_MAC             = $WM;/" $TB
    python3 $TBDIR/gen_outlier_rtl_check.py $IB $F $WM > /dev/null
    rm -rf xcelium.d worklib
    xrun -clean -f $TBDIR/tr_nonlinear_vpu.f > xrun_outlier_${NAME}.log
    echo "=== ${NAME} (W_VEC=$W FRAC_W=$F W_MAC=$WM) ==="
    python3 $TBDIR/compare_outlier.py
end
rm -f meta.txt
sed -i "s/localparam int W_VEC             = [0-9]*;/localparam int W_VEC             = 8;/" $TB
sed -i "s/localparam int FRAC_W            = [0-9]*;/localparam int FRAC_W            = 4;/" $TB
sed -i "s/localparam int RM_CONST_LN_SQRT_N = [0-9]*;/localparam int RM_CONST_LN_SQRT_N = 17;/" $TB
sed -i "s/localparam int W_MAC             = [0-9]*;/localparam int W_MAC             = 16;/" $TB
