################################################################################
#
# Init setup file
# Created by Genus(TM) Synthesis Solution on 03/10/2026 21:32:35
#
################################################################################
if { ![is_common_ui_mode] } { error "ERROR: This script requires common_ui to be active."}

read_netlist /project/test_project/users/yonatap/ws/tr-vit/workspace/../export/post_synth/softmax_engine.v

init_design
