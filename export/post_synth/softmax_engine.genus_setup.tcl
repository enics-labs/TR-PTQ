################################################################################
#
# Genus(TM) Synthesis Solution setup file
# Created by Genus(TM) Synthesis Solution 25.12-s067_1
#   on 03/10/2026 21:32:35
#
# This file can only be run in Genus Common UI mode.
#
################################################################################


# This script is intended for use with Genus(TM) Synthesis Solution version 25.12-s067_1


# Remove Existing Design
################################################################################
if {[::legacy::find -design design:softmax_engine] ne ""} {
  puts "** A design with the same name is already loaded. It will be removed. **"
  delete_obj design:softmax_engine
}


# To allow user-readonly attributes
################################################################################
::legacy::set_attribute -quiet force_tui_is_remote 1 /

phys::read_script /project/test_project/users/yonatap/ws/tr-vit/workspace/../export/post_synth/softmax_engine.root.g


# Libraries
################################################################################
::legacy::set_attribute library {/data/tsmc/65LP/dig_libs/ARM_FEONLY/arm/tsmc/cln65lp//sc9_base_rvt/r0p0/lib/sc9_cln65lp_base_rvt_ss_typical_max_0p90v_125c.lib /data/tsmc/65LP/dig_libs/ARM_FEONLY/arm/tsmc/cln65lp//sc9_base_lvt/r0p0/lib/sc9_cln65lp_base_lvt_ss_typical_max_0p90v_125c.lib /data/tsmc/65LP/dig_libs/ARM_FEONLY/arm/tsmc/cln65lp//sc9_base_hvt/r0p0/lib/sc9_cln65lp_base_hvt_ss_typical_max_0p90v_125c.lib} /


# Design
################################################################################
read_netlist -top softmax_engine /project/test_project/users/yonatap/ws/tr-vit/workspace/../export/post_synth/softmax_engine.v
read_metric -id current /project/test_project/users/yonatap/ws/tr-vit/workspace/../export/post_synth/softmax_engine.metrics.json

phys::read_script /project/test_project/users/yonatap/ws/tr-vit/workspace/../export/post_synth/softmax_engine.g

phys::read_lec_taf /project/test_project/users/yonatap/ws/tr-vit/workspace/../export/post_synth/softmax_engine.lec.taf.gz
phys::read_taf /project/test_project/users/yonatap/ws/tr-vit/workspace/../export/post_synth/softmax_engine.safety.taf.gz
puts "\n** Restoration Completed **\n"


# Data Integrity Check
################################################################################
# program version
if {"[string_representation [::legacy::get_attribute program_version /]]" != "25.12-s067_1"} {
   mesg_send [::legacy::find -message /messages/PHYS/PHYS-91] "golden program_version: 25.12-s067_1  current program_version: [string_representation [::legacy::get_attribute program_version /]]"
}
# license
if {"[string_representation [::legacy::get_attribute startup_license /]]" != "Genus_Synthesis"} {
   mesg_send [::legacy::find -message /messages/PHYS/PHYS-91] "golden license: Genus_Synthesis  current license: [string_representation [::legacy::get_attribute startup_license /]]"
}
# slack
set _slk_ [::legacy::get_attribute slack design:softmax_engine]
if {[regexp {^-?[0-9.]+$} $_slk_]} {
  set _slk_ [format %.1f $_slk_]
}
if {$_slk_ != "94.1"} {
   mesg_send [::legacy::find -message /messages/PHYS/PHYS-92] "golden slack: 94.1,  current slack: $_slk_"
}
unset _slk_
# multi-mode slack
# tns
set _tns_ [::legacy::get_attribute tns design:softmax_engine]
if {[regexp {^-?[0-9.]+$} $_tns_]} {
  set _tns_ [format %.0f $_tns_]
}
if {$_tns_ != "0"} {
   mesg_send [::legacy::find -message /messages/PHYS/PHYS-92] "golden tns: 0,  current tns: $_tns_"
}
unset _tns_
# cell area
set _cell_area_ [::legacy::get_attribute cell_area design:softmax_engine]
if {[regexp {^-?[0-9.]+$} $_cell_area_]} {
  set _cell_area_ [format %.0f $_cell_area_]
}
if {$_cell_area_ != "27819"} {
   mesg_send [::legacy::find -message /messages/PHYS/PHYS-92] "golden cell area: 27819,  current cell area: $_cell_area_"
}
unset _cell_area_
# net area
set _net_area_ [::legacy::get_attribute net_area design:softmax_engine]
if {[regexp {^-?[0-9.]+$} $_net_area_]} {
  set _net_area_ [format %.0f $_net_area_]
}
if {$_net_area_ != "0"} {
   mesg_send [::legacy::find -message /messages/PHYS/PHYS-92] "golden net area: 0,  current net area: $_net_area_"
}
unset _net_area_
