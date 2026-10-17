# TR-PTQ VPU: RTL

RTL for a Taylor-Region (TR) vector processing unit: one shared, software-defined log/exp crossbar
(`tr_nonlinear_vpu`) that implements the nonlinear operators of INT8 Llama-3.2-1B inference, sequenced by an MMIO
command FSM and fed by an INT8 matmul engine.

| Operator | Used by | Where it runs |
|---|---|---|
| RMSNorm (rows up to thousands of elements, host-sequenced in 8-lane tiles) | Llama | VPU |
| SwiGLU (silu x up, two-pass per-row gain, rows up to 8192) | Llama | VPU |
| Softmax (one N=8 tile with its `z = max + ln S`; flash-attention orchestration on the CPU) | Llama | VPU |

This is `tr-vpu_rtl`, the RTL-only branch of [TR-PTQ](https://github.com/enics-labs/TR-PTQ).
The Genus synthesis flow is on the `tr-vpu_synthesis` branch. The paper-era design (ViT, GELU, single-tile RMSNorm,
the I-BERT and TR baselines) is frozen at the tag `tr-vpu_rtl-ISCAS-20261013`; this branch carries none of it.

Start with [docs/architecture.md](docs/architecture.md).

## Layout

| Path | Content |
|---|---|
| `rtl_soc/` | production RTL: `tr_nonlinear_vpu` (the crossbar), `tr_soc_ctrl` (command FSMs), `tr_soc_top_int` / `tr_soc_top_mx` (integration), and the shared engines |
| `tb_soc/` | testbenches, mirroring the RTL tree |
| `verification/` | `verify_block.py`, the golden-vector regression flow |
| `emulation/` | C++ golden models (`tr_math_model.hpp`, `cpu_math_model.cpp`) and the ctypes wrapper source |
| `scripts/` | `xrun` configuration shared by all file lists |
| `docs/` | architecture, verification, per-module reference |
| `workspace/` | run everything from here |

## Running

```sh
cd workspace
xrun -f ../tb_soc/<block>/<block>.f                      # self-checking testbench
python3 ../verification/verify_block.py <block>          # RTL against the C++ golden model
```

[docs/verification.md](docs/verification.md) lists every block, its golden model and the energy pilots.

## Documentation

| Page | Content |
|---|---|
| [docs/architecture.md](docs/architecture.md) | shared crossbar, command set, timing structure, INT and MX variants |
| [docs/verification.md](docs/verification.md) | regression flows, golden models, energy pilots |
| [docs/module_reference/](docs/module_reference/) | ports and parameters, one file per `rtl_soc/` subfolder |
| [docs/mx_format_operation.md](docs/mx_format_operation.md) | the MX (shared-exponent) datapath |
| `docs/*.drawio` | block diagram sources |
