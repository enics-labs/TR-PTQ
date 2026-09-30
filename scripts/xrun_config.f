// simulation flags
-sv
// -uvm // not for now
-log logs/run.log
-access +rwc

// Standard UVM library
// +define+UVM_NO_DEPRECATED 

// simulation parameters
-timescale 1ns/1ps
-input ../scripts/simulation.tcl
