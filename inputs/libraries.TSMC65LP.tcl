# Technology settings for the TSMC 65LP Technology
#    Note that standard cell library setting are in a separate file
#    to support various standard cell libraries
#
set METAL_STACK 1p9m_6x2z
set TRACKS 9 
set_db design_process_node 65

#set paths(PDK_ROOT) $env(kit_dir)/../
set paths(PDK_ROOT) /data/tsmc/65LP

# Technology Setup
set paths(ARM_ROOT) $paths(PDK_ROOT)/dig_libs/ARM_FEONLY/arm/tsmc/cln65lp/
set paths(TECHNOLOGY_FILES) $paths(ARM_ROOT)/arm_tech/r2p0

# Parasitic Extraction 
set tech_files(CAPTABLE_BC) $paths(TECHNOLOGY_FILES)/cadence_captable/${METAL_STACK}/rcbest.captbl
set tech_files(CAPTABLE_TC) $paths(TECHNOLOGY_FILES)/cadence_captable/${METAL_STACK}/typical.captbl
set tech_files(CAPTABLE_WC) $paths(TECHNOLOGY_FILES)/cadence_captable/${METAL_STACK}/rcworst.captbl
set paths(QRC_ROOT) $paths(PDK_ROOT)/QRC/1.3a/
set tech_files(QRCTECH_FILE_TYPICAL) $paths(QRC_ROOT)/typical/qrcTechFile
set tech_files(QRCTECH_FILE_CBEST) $paths(QRC_ROOT)/cbest/qrcTechFile
set tech_files(QRCTECH_FILE_CWORST) $paths(QRC_ROOT)/cworst/qrcTechFile
set tech_files(QRCTECH_FILE_RCBEST) $paths(QRC_ROOT)/rcbest/qrcTechFile
set tech_files(QRCTECH_FILE_RCWORST) $paths(QRC_ROOT)/rcworst/qrcTechFile
set tech_files(QRCTECH_FILE_BC) $tech_files(QRCTECH_FILE_RCBEST)
set tech_files(QRCTECH_FILE_TC) $tech_files(QRCTECH_FILE_TYPICAL)
set tech_files(QRCTECH_FILE_WC) $tech_files(QRCTECH_FILE_RCWORST)

# TECHLEF
# Technology LEF is defined in the standard cell LEF
#   set tech_files(TECHNOLOGY_LEF) ""
#        set tech_files(ALL_LEFS) [list $tech_files(TECHNOLOGY_LEF)] ; # list of all lefs for init_design

# DRC Related
set tech(layer_names) "M0 M1 M2 M3 M4 M5 M6 M7 M8 M9"
set tech(row_height) 1.8 
set tech(grid_unit) 0.2
#set tech_files(layer_names) [lrange  [dbGet head.layers.name] 1 11]
#set tech_files(min_spacing_x) [lindex  [dbGet head.layers.minSpacing] 1]
#set tech_files(min_width_x) [lindex  [dbGet head.layers.minWidth] 1]
#set tech_files(min_spacing_y) [lindex  [dbGet head.layers.minSpacing] 2]
#set tech_files(min_width_y) [lindex  [dbGet head.layers.minWidth] 2]
#set tech_files(min_spacing_stripes) 0.25 ; # Comes from [dbGet head.layers.spacingTables]


# Routing Rules
#create_route_rule -width {Metal1 0.12 Metal2 0.14 Metal3 0.14 Metal4 0.14 Metal5 0.14 Metal6 0.14 Metal7 0.14 Metal8 0.14 Metal9 0.14 } \
#		-spacing {Metal1 0.12 Metal2 0.14 Metal3 0.14 Metal4 0.14 Metal5 0.14 Metal6 0.14 Metal7 0.14 Metal8 0.14 Metal9 0.14 } -name 2w2s
#create_route_type -name clkroute -route_rule 2w2s -bottom_preferred_layer Metal5 -top_preferred_layer Metal6
#set_db cts_route_type_trunk clkroute
#set_db cts_route_type_leaf clkroute
#set_db cts_route_type_leaf    leaf
#  set_db cts_route_type_trunk   trunk
#  set_db cts_route_type_top     top
#set_db cts_buffer_cells {CLKBUFX8 CLKBUFX12}
#set_db cts_clock_gating_cells TLATNTSCA
#  create_route_type -name leaf \
#    -top_preferred_layer    Metal4 \
#    -bottom_preferred_layer Metal3
#  create_route_type -name trunk \
#    -top_preferred_layer    Metal6 \
#    -bottom_preferred_layer Metal5
#  create_route_type -name top \
#    -top_preferred_layer    Metal6 \
#    -bottom_preferred_layer Metal5

#############################################
#       Print values to debug file
#############################################
set df [open $debug_file a]
puts $df "\n******************************************"
puts $df "* Values loaded from libraries.$TECHNOLOGY.tcl *"
puts $df "******************************************"
foreach var {METAL_STACK TRACKS} {
    puts $df "$var = \t[set $var]"
}

foreach dic {paths tech tech_files} {
    foreach key [array names $dic] {
        puts $df "${dic}(${key}) = \t[set ${dic}([set key])]"
    }
}
close $df