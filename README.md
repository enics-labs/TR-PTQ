# SOC requirements
## External DRAM access (DDR)

Parameters / activations must fit in off-chip DRAM.

This dominates the rest of the architecture: you either
a. hang on vendor DDR controller IP, or
b. use an existing SoC framework that already does this.

# On-chip SRAM 
Cache for spatial locality.

Software-managed SRAM (or accelerator-managed) to:

* Load tiles / blocks from DRAM

* Reuse them heavily in compute engines (MAC arrays etc.)

* Likely multi-bank SRAM attached closely to the accelerator.

## DMA engine for burst transfers between DDR ? SRAM ? accelerator

Offload CPU from doing memcpy loops.

DMA must:

* Speak AXI (or similar) to DDR

* Offer linear / strided / 2D transfers

* Possibly chainable descriptors.

## Small controller CPU

Role:

* Program the DMA and accelerators.

* Manage scheduling, tiling, command lists, maybe an RTOS.

Requirements:

* Does not need to be application-class.

* Must have easy integration to AXI / interconnect.

* Tooling must be sane (GCC, debug, etc.).

## Small instruction memory for the controller

Could be:

* On-chip instruction TCM (ITCM) tightly coupled to CPU, or

* Simple SRAM behind the interconnect.

Size is modest (tens of KB to a few hundred KB).

## Simple control ISA / software stack

* Bare-metal C or tiny RTOS.

* No need for Linux, MMU, or complex privilege levels.

## Basic peripheral set
UART:

* For boot, debug, logging, maybe simple command protocol.

SPI:

* For low-speed off-chip devices (flash, sensors) or external control.

Ethernet (nice-to-have but important):

* For high-level control plane or streaming test data.

* Great for a ?remote accelerator card? connected over the network.

