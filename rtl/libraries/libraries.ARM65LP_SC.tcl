# Library setting for ARM Standard Cell libraries for TSMC65LP technology

# Libs
#set paths(ARM_ROOT)		$paths(PDK_ROOT)/dig_libs/ARM_FEONLY/arm/tsmc/cln65lp
set paths(STANDARD_CELLS_RVT)	$paths(ARM_ROOT)/sc${TRACKS}_base_rvt/r0p0
lappend paths(LIB_paths) "$paths(STANDARD_CELLS_RVT)/lib/"
set paths(STANDARD_CELLS_LVT)	$paths(ARM_ROOT)/sc${TRACKS}_base_lvt/r0p0
lappend paths(LIB_paths) "$paths(STANDARD_CELLS_LVT)/lib/"
set paths(STANDARD_CELLS_HVT)	$paths(ARM_ROOT)/sc${TRACKS}_base_hvt/r0p0
lappend paths(LIB_paths) "$paths(STANDARD_CELLS_HVT)/lib/"

# General
set tech(LIBRARY_HAS_ENDCAPS) "YES"
set tech(STANDARD_CELL_SITE) sc9_cln65lp
set tech(STANDARD_CELL_VDD) VDD
set tech(STANDARD_CELL_GND) VSS

# LEFS
set tech(SC_LEF_SUPPRESS_MESSAGES_GENUS) "PHYS-279 PHYS-129 PHYS-15 PHYS-12 LBR-162 LBR-155"
set tech(SC_LEF_SUPPRESS_MESSAGES_INNOVUS) "IMPLF-200"
set tech_files(TECHNOLOGY_LEF) $paths(TECHNOLOGY_FILES)/lef/${METAL_STACK}/sc${TRACKS}_tech.lef
    set tech_files(ALL_LEFS) [list $tech_files(TECHNOLOGY_LEF)] ; # list of all lefs for init_design
set tech_files(STANDARD_CELLS_RVT_LEF) $paths(STANDARD_CELLS_RVT)/lef/sc${TRACKS}_cln65lp_base_rvt.lef
    lappend tech_files(ALL_LEFS) $tech_files(STANDARD_CELLS_RVT_LEF)
set tech_files(STANDARD_CELLS_LVT_LEF) $paths(STANDARD_CELLS_LVT)/lef/sc${TRACKS}_cln65lp_base_lvt.lef
    lappend tech_files(ALL_LEFS) $tech_files(STANDARD_CELLS_LVT_LEF)
set tech_files(STANDARD_CELLS_HVT_LEF) $paths(STANDARD_CELLS_HVT)/lef/sc${TRACKS}_cln65lp_base_hvt.lef
    lappend tech_files(ALL_LEFS) $tech_files(STANDARD_CELLS_HVT_LEF)

# Temperatures for Corners
set tech(TEMPERATURE_BC) -40
set tech(TEMPERATURE_TC) 25
set tech(TEMPERATURE_WC) 125

# Libs
set tech(SC_LIB_SUPPRESS_MESSAGES_GENUS) "LBR-9 LBR-76 LBR-40 LBR-436 LBR-170"
set tech(SC_LIB_SUPPRESS_MESSAGES_INNOVUS) "TECHLIB-436 TECHLIB-1318 TECHLIB-302"
set tech_files(STANDARD_CELLS_RVT_BC_LIB) $paths(STANDARD_CELLS_RVT)/lib/sc${TRACKS}_cln65lp_base_rvt_ff_typical_min_1p32v_m40c.lib
    lappend tech_files(ALL_BC_LIBS) [list $tech_files(STANDARD_CELLS_RVT_BC_LIB)] ; # list of all libs for init_design
set tech_files(STANDARD_CELLS_RVT_WC_LIB) $paths(STANDARD_CELLS_RVT)/lib/sc${TRACKS}_cln65lp_base_rvt_ss_typical_max_0p90v_125c.lib
    lappend tech_files(ALL_WC_LIBS) [list $tech_files(STANDARD_CELLS_RVT_WC_LIB)] ; # list of all libs for init_design
set tech_files(STANDARD_CELLS_RVT_TC_LIB) $paths(STANDARD_CELLS_RVT)/lib/sc${TRACKS}_cln65lp_base_rvt_tt_typical_max_1p00v_25c.lib
    lappend tech_files(ALL_TC_LIBS) [list $tech_files(STANDARD_CELLS_RVT_TC_LIB)] ; # list of all libs for init_design

set tech_files(STANDARD_CELLS_LVT_BC_LIB) $paths(STANDARD_CELLS_LVT)/lib/sc${TRACKS}_cln65lp_base_lvt_ff_typical_min_1p32v_m40c.lib
    lappend tech_files(ALL_BC_LIBS) [list $tech_files(STANDARD_CELLS_LVT_BC_LIB)] ; # list of all libs for init_design
set tech_files(STANDARD_CELLS_LVT_WC_LIB) $paths(STANDARD_CELLS_LVT)/lib/sc${TRACKS}_cln65lp_base_lvt_ss_typical_max_0p90v_125c.lib
    lappend tech_files(ALL_WC_LIBS) [list $tech_files(STANDARD_CELLS_LVT_WC_LIB)] ; # list of all libs for init_design
set tech_files(STANDARD_CELLS_LVT_TC_LIB) $paths(STANDARD_CELLS_LVT)/lib/sc${TRACKS}_cln65lp_base_lvt_tt_typical_max_1p00v_25c.lib
    lappend tech_files(ALL_TC_LIBS) [list $tech_files(STANDARD_CELLS_LVT_TC_LIB)] ; # list of all libs for init_design

set tech_files(STANDARD_CELLS_HVT_BC_LIB) $paths(STANDARD_CELLS_HVT)/lib/sc${TRACKS}_cln65lp_base_hvt_ff_typical_min_1p32v_m40c.lib
    lappend tech_files(ALL_BC_LIBS) [list $tech_files(STANDARD_CELLS_HVT_BC_LIB)] ; # list of all libs for init_design
set tech_files(STANDARD_CELLS_HVT_WC_LIB) $paths(STANDARD_CELLS_HVT)/lib/sc${TRACKS}_cln65lp_base_hvt_ss_typical_max_0p90v_125c.lib
    lappend tech_files(ALL_WC_LIBS) [list $tech_files(STANDARD_CELLS_HVT_WC_LIB)] ; # list of all libs for init_design
set tech_files(STANDARD_CELLS_HVT_TC_LIB) $paths(STANDARD_CELLS_HVT)/lib/sc${TRACKS}_cln65lp_base_hvt_tt_typical_max_1p00v_25c.lib
    lappend tech_files(ALL_TC_LIBS) [list $tech_files(STANDARD_CELLS_HVT_TC_LIB)] ; # list of all libs for init_design

  

# Verilog
set tech_files(STANDARD_CELLS_RVT_VERILOG) $paths(STANDARD_CELLS_RVT)/verilog/sc9_cln65lp_base_rvt.v
    lappend tech_files(ALL_BEHAVIORAL_MODELS) [list $tech_files(STANDARD_CELLS_RVT_VERILOG)]
set tech_files(STANDARD_CELLS_LVT_VERILOG) $paths(STANDARD_CELLS_LVT)/verilog/sc9_cln65lp_base_lvt.v
    lappend tech_files(ALL_BEHAVIORAL_MODELS) [list $tech_files(STANDARD_CELLS_LVT_VERILOG)]
set tech_files(STANDARD_CELLS_HVT_VERILOG) $paths(STANDARD_CELLS_HVT)/verilog/sc9_cln65lp_base_hvt.v
    lappend tech_files(ALL_BEHAVIORAL_MODELS) [list $tech_files(STANDARD_CELLS_HVT_VERILOG)]

# OA
set tech_files(STANDARD_CELLS_RVT_OA) $paths(STANDARD_CELLS_RVT)/oa
set tech_files(STANDARD_CELLS_LVT_OA) $paths(STANDARD_CELLS_LVT)/oa
set tech_files(STANDARD_CELLS_HVT_OA) $paths(STANDARD_CELLS_HVT)/oa

# Annoying messages



# For SDC
set tech(SDC_DRIVING_CELL) BUF_X6M_A9TR
set tech(CCOPT_DRIVING_PIN) BUF_X6M_A9TR/Y
set tech(SDC_LOAD_PIN) BUF_X6M_A9TR/A


# Physical Cells
set tech(WELLTAP) "FILLTIE2_A9TR"; #"WELLANTENNATIEPW2_A9TR"
set tech(WELLTAP_RULE) 20.0
set tech(ENDCAPS) "ENDCAPTIE2_A9TR ENDCAPTIE2_A9TR"
set tech(TIEHI) "TIEHI_X1M_A9TR"
set tech(TIELO) "TIELO_X1M_A9TR"
set tech(FILLERS) "FILL128_A9TR FILL64_A9TR FILL32_A9TR FILL16_A9TR FILL8_A9TR FILL4_A9TR FILL2_A9TR FILL1_A9TR" 
set tech(FILLTIE) "FILLTIE128_A9TR FILLTIE64_A9TR FILLTIE32_A9TR FILLTIE16_A9TR FILLTIE8_A9TR FILLTIE4_A9TR FILLTIE2_A9TR"
set tech(DECAP) "FILLCAP128_A9TR FILLCAP64_A9TR FILLCAP32_A9TR FILLCAP16_A9TR FILLCAP8_A9TR FILLCAP4_A9TR FILLCAP3_A9TR" 
set tech(DECAPTIE) "FILLCAPTIE128_A9TR FILLCAPTIE64_A9TR FILLCAPTIE32_A9TR FILLCAPTIE16_A9TR FILLCAPTIE8_A9TR FILLCAPTIE6_A9TR" 
set tech(ANTENNA_DIODE) "ANTENNA2_A9TR"
if {$tech(LIBRARY_HAS_ENDCAPS)=="YES"} {
    set tech(ENDCAPS) "ENDCAPTIE2_A9TR ENDCAPTIE2_A9TR"
}

# Clock Cells (Used during CTS/CCOpt)
set tech(CLOCK_BUFFERS) "BUF_X16B_A9TR BUF_X13B_A9TR BUF_X11B_A9TR BUF_X9B_A9TR BUF_X4B_A9TR"
set tech(CLOCK_INVERTERS) "INV_X16B_A9TR INV_X13B_A9TR INV_X11B_A9TR INV_X9B_A9TR INV_X5B_A9TR INV_X3B_A9TR"
set tech(CLOCK_GATES) "POSTICG_X9B_A9TR POSTICG_X6B_A9TR POSTICG_X4B_A9TR POSTICG_X1P4B_A9TR POSTICG_X1B_A9TR"
set tech(CLOCK_LOGIC) "MX2_X8B_A9TR MX2_X4B_A9TR NAND2_X8B_A9TR NAND2_X4B_A9TR NOR2_X8B_A9TR NOR2_X4B_A9TR"
set tech(CLOCK_DELAYS) "DLY4_X4M_A9TR DLY4_X2M_A9TR DLY4_X1M_A9TR DLY4_X0P5M_A9TR DLY2_X4M_A9TR DLY2_X2M_A9TR DLY2_X1M_A9TR DLY2_X0P5M_A9TR"

# Technology specific settings:
set_db route_design_with_via_in_pin true ; # Defined in ARM SC user scripts


#############################################
#       Print values to debug file
#############################################
set df [open $debug_file a]
puts $df "\n******************************************"
puts $df "* Values loaded after libraries.$SC_TECHNOLOGY.tcl *"
puts $df "******************************************"


foreach dic {paths tech tech_files} {
    foreach key [array names $dic] {
        puts $df "${dic}(${key}) = \t[set ${dic}([set key])]"
    }
}

close $df
