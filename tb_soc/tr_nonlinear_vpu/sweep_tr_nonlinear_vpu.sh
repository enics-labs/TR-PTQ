#!/bin/tcsh
# Run from workspace/: tcsh ../tb_soc/tr_nonlinear_vpu/sweep_tr_nonlinear_vpu.sh
# Sweeps tr_nonlinear_vpu's full GELU/Softmax/RMSNorm(decay)/RMSNorm(growth)
# sequences across all 6 target formats at the FROZEN configuration
# (anchor table Q0.8/LUT_IDX_W=3 everywhere, W_MAC per format below: only Q4.8 needs 18),
# comparing RTL output to the Python golden model (tools/scripts/vpu_precision/tr_vpu_model.py,
# native I/O mode; bit-exact to tr_math_model.hpp at Q4.4).
setenv CDS_LIC_FILE 5280@enicsw01
setenv DISPLAY ""

set TBDIR = ../tb_soc/tr_nonlinear_vpu
set TB = $TBDIR/tr_nonlinear_vpu_tb.sv

# name:W_VEC:FRAC_W:int_bits:RM_CONST_LN_SQRT_N:W_MAC
# RM_CONST_LN_SQRT_N = round(17/16 * 2^FRAC_W) = 17 << (FRAC_W-4)
foreach fmt (Q4.4:8:4:4:17:16 Q4.6:10:6:4:68:16 Q4.8:12:8:4:272:18 Q6.4:10:4:6:17:16 Q8.4:12:4:8:17:16 Q12.4:16:4:12:17:16)
    set parts = `echo $fmt | tr ':' ' '`
    set NAME = $parts[1]
    set W = $parts[2]
    set F = $parts[3]
    set IB = $parts[4]
    set C = $parts[5]
    set WM = $parts[6]

    echo "=== $NAME (W_VEC=$W FRAC_W=$F RM_CONST=$C W_MAC=$WM) ==="

    sed -i "s/localparam int W_VEC             = [0-9]*;/localparam int W_VEC             = $W;/" $TB
    sed -i "s/localparam int FRAC_W            = [0-9]*;/localparam int FRAC_W            = $F;/" $TB
    sed -i "s/localparam int RM_CONST_LN_SQRT_N = [0-9]*;/localparam int RM_CONST_LN_SQRT_N = $C;/" $TB
    sed -i "s/localparam int W_MAC             = [0-9]*;/localparam int W_MAC             = $WM;/" $TB

    python3 $TBDIR/gen_vpu_golden.py $IB $F $WM

    rm -rf xcelium.d worklib
    xrun -clean -f $TBDIR/tr_nonlinear_vpu.f > xrun_vpu_${NAME}.log
    if ($status != 0) then
        echo "  [FATAL] xrun failed, see xrun_vpu_${NAME}.log"
        continue
    endif

    python3 $TBDIR/compare_vpu.py
end

# leave the testbench at its Q4.4 defaults
sed -i "s/localparam int W_VEC             = [0-9]*;/localparam int W_VEC             = 8;/" $TB
sed -i "s/localparam int FRAC_W            = [0-9]*;/localparam int FRAC_W            = 4;/" $TB
sed -i "s/localparam int RM_CONST_LN_SQRT_N = [0-9]*;/localparam int RM_CONST_LN_SQRT_N = 17;/" $TB
sed -i "s/localparam int W_MAC             = [0-9]*;/localparam int W_MAC             = 16;/" $TB
