################################################################################
#       Setting dont_touch and dont_use preserve setting
################################################################################
catch {set_db hinst:softmax_engine/RC_CG_HIER_INST0 .dont_touch_hports true}
catch {set_db hinst:softmax_engine/dsp_mac/RC_CG_HIER_INST1 .dont_touch_hports true}
catch {set_db hinst:softmax_engine/dsp_mac/RC_CG_HIER_INST2 .dont_touch_hports true}
