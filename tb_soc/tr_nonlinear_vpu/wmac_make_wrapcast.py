import sys
s = open(sys.argv[1]).read()
old = "bb_shift_mode ? rms_s_sat :"
assert old in s, "saturating cast not found in RTL"
open("tr_nonlinear_vpu_wrapcast.sv", "w").write(s.replace(old, "bb_shift_mode ? W_MAC'(vpu_dot_out >>> FRAC_W) :"))
