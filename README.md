# TR-PTQ VPU — RTL

RTL for a Taylor-Region (TR) Vector Processing Unit: a single shared, software-defined log/exp crossbar (`tr_nonlinear_vpu`) that implements GELU, Softmax, and RMSNorm for INT8 ViT inference, driven by an MMIO command FSM, alongside the linear (matmul) engine that feeds it. Includes I-BERT-style and standalone-TR baseline RTL for energy/area comparison.

This is `tr-vpu_rtl`, the RTL-only branch of [TR-PTQ](https://github.com/enics-labs/TR-PTQ). The Genus (65nm) synthesis flow lives on the `tr-vpu_synthesis` branch instead.

**Start here:** [docs/architecture.md](docs/architecture.md) for the design overview, or [docs/tr_soc_architecture.drawio](docs/tr_soc_architecture.drawio) for the block diagram.

## Directory structure

```
rtl_soc/              Production RTL: tr_nonlinear_vpu (the shared crossbar),
                       tr_soc_ctrl (the FSMs that sequence it), tr_soc_top_int/mx
                       (top-level SoC integration), and the shared engines
                       (mult_engines, dot_product_engine, requantize_engine,
                       mx_modules) they're built from.
rtl_baseline/          Standalone RTL kept for comparison, NOT instantiated by
                       either SoC top: tr_baseline/ (dedicated TR modules) and
                       ibert_baseline/ (I-BERT-style comparison point).
tb_soc/                Testbenches for rtl_soc/, mirroring its structure 1:1.
tb_baseline/            Testbenches for rtl_baseline/, same mirroring.
verification/          verify_block.py -- the golden-vector verification flow.
emulation/             cpu_math_model.cpp / tr_math_model.hpp -- the C++ golden
                       models verify_block.py checks RTL against.
scripts/               xrun config shared by every .f file.
docs/                  Architecture, module reference, verification docs.
workspace/              Run everything from here (xrun, verify_block.py).
```

## Running things

Every block has an `.f` file (an `xrun` file list). From `workspace/`:

```sh
cd workspace
xrun -f ../tb_soc/<block>/<block>.f
```

For blocks verified against the C++ golden model instead of a self-checking testbench:

```sh
cd workspace
python3 ../verification/verify_block.py <block_name>
```

See [docs/verification.md](docs/verification.md) for the full block-name table and how the golden-vector flow works.

## Docs

- [docs/architecture.md](docs/architecture.md) — design overview: the shared crossbar, INT vs MX datapath variants, where the baseline RTL fits in.
- [docs/mx_format_operation.md](docs/mx_format_operation.md) — the MX (Microscaling shared-exponent) datapath in detail.
- [docs/verification.md](docs/verification.md) — how to run and verify anything in this repo.
- [docs/module_reference/](docs/module_reference/) — per-block port/parameter reference, one file per `rtl_soc/` subfolder.
- [docs/tr_soc_architecture.drawio](docs/tr_soc_architecture.drawio) / [docs/research_diagrams.drawio](docs/research_diagrams.drawio) — diagram sources.
