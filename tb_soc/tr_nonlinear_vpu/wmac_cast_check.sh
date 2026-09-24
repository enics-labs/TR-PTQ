#!/bin/tcsh
# Run from workspace/: tcsh ../tb_soc/tr_nonlinear_vpu/wmac_cast_check.sh
# For every format: run identical inputs through (a) the patched RTL
# (saturating W_MAC cast) and (b) a copy of it with the old wrapping cast
# (built on the fly into workspace/, repo RTL untouched), and compare both to
# the golden model (saturating). Expected: ops 0/1/2/3 identical in (a) and
# (b) and equal to golden; op 5 (S>>FRAC_W >= 2^W_MAC) equal to golden in (a)
# and different in (b). The last config is a stress case (Q4.8 at W_MAC=16, 
# formats that need more) where the cast fires on many ordinary vectors.
setenv CDS_LIC_FILE 5280@enicsw01
setenv DISPLAY ""
set TBDIR = ../tb_soc/tr_nonlinear_vpu
set TB = $TBDIR/tr_nonlinear_vpu_tb.sv
set VPU = ../rtl_soc/tr_nonlinear_vpu/tr_nonlinear_vpu.sv

python3 $TBDIR/wmac_make_wrapcast.py $VPU
sed "s#../rtl_soc/tr_nonlinear_vpu/tr_nonlinear_vpu.sv#tr_nonlinear_vpu_wrapcast.sv#" $TBDIR/tr_nonlinear_vpu.f > tr_nonlinear_vpu_wrapcast.f

# name:W_VEC:FRAC_W:int_bits:RM_CONST:W_MAC
foreach fmt (Q4.4:8:4:4:17:16 Q4.6:10:6:4:68:16 Q4.8:12:8:4:272:18 Q6.4:10:4:6:17:16 Q8.4:12:4:8:17:16 Q12.4:16:4:12:17:16 Q4.8@16:12:8:4:272:16)
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
    python3 $TBDIR/gen_vpu_golden.py $IB $F $WM > /dev/null
    rm -rf xcelium.d worklib
    xrun -clean -f $TBDIR/tr_nonlinear_vpu.f > xrun_cast_post_${NAME}.log
    cp hdl_out.txt hdl_out_post.txt
    rm -rf xcelium.d worklib
    xrun -clean -f tr_nonlinear_vpu_wrapcast.f > xrun_cast_pre_${NAME}.log
    cp hdl_out.txt hdl_out_pre.txt
    echo "=== ${NAME} (W_VEC=$W FRAC_W=$F W_MAC=$WM) ==="
    python3 $TBDIR/wmac_cast_check.py
end

rm -f tr_nonlinear_vpu_wrapcast.sv tr_nonlinear_vpu_wrapcast.f hdl_out_pre.txt hdl_out_post.txt
sed -i "s/localparam int W_VEC             = [0-9]*;/localparam int W_VEC             = 8;/" $TB
sed -i "s/localparam int FRAC_W            = [0-9]*;/localparam int FRAC_W            = 4;/" $TB
sed -i "s/localparam int RM_CONST_LN_SQRT_N = [0-9]*;/localparam int RM_CONST_LN_SQRT_N = 17;/" $TB
sed -i "s/localparam int W_MAC             = [0-9]*;/localparam int W_MAC             = 16;/" $TB
