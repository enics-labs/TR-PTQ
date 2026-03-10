################################################################################
#
# Init setup file
# Created by Genus(TM) Synthesis Solution on 03/10/2026 19:23:35
#
################################################################################
if { ![is_common_ui_mode] } { error "ERROR: This script requires common_ui to be active."}

read_netlist /project/test_project/users/yonatap/ws/tr-vit/workspace/../export/post_elaboartion/softmax_engine.v

init_design
