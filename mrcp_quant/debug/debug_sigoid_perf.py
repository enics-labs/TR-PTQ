import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from mrcp_quant.quant_utils import TaylorExponent, new_ln


TensorFn = Callable[[torch.Tensor], torch.Tensor]
TRPTQ_GELU = None
TRPTQ_GELU_IMPORT_ERROR = None


def load_trptq_gelu():
    global TRPTQ_GELU, TRPTQ_GELU_IMPORT_ERROR
    if TRPTQ_GELU is not None or TRPTQ_GELU_IMPORT_ERROR is not None:
        return TRPTQ_GELU

    gelu_dir = Path(__file__).resolve().parents[1] / "optimized_layers" / "gelu"
    if str(gelu_dir) not in sys.path:
        sys.path.append(str(gelu_dir))

    try:
        import trptq_gelu
    except Exception as exc:
        TRPTQ_GELU_IMPORT_ERROR = exc
        return None

    TRPTQ_GELU = trptq_gelu
    return TRPTQ_GELU


def cuda_optimized_skip_reason(cfg: "BenchConfig") -> Optional[str]:
    if cfg.device.type != "cuda":
        return "requires CUDA"
    if cfg.dtype != torch.float32:
        return "extension expects float32 inputs"
    if load_trptq_gelu() is None:
        return str(TRPTQ_GELU_IMPORT_ERROR)
    return None


@dataclass(frozen=True)
class BenchConfig:
    alpha: float
    nof_bits: int
    lut_size: int
    split_table: int
    iterations: int
    warmup: int
    repeats: int
    device: torch.device
    dtype: torch.dtype


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def float_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def sigmoid_gelu(x: torch.Tensor, alpha: float = 1.702) -> torch.Tensor:
    return x * torch.sigmoid(x * alpha)


def int_sigmoid_exp_terms(
    x: torch.Tensor,
    nof_bits: int = 16,
    alpha: float = 1.702,
    lut_size: int = 16,
    split_table: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror IntGeluTS.int_sigmoid without instantiating the CUDA-only module."""
    input_bits = nof_bits - 4
    output_bits = nof_bits

    x_sig = x * alpha
    x_max = torch.clamp(x_sig, min=0)
    x_int = x_sig - x_max

    spacial_scale = 1 << (input_bits - 1)
    x_scale = (x_int * spacial_scale).floor().to(dtype=torch.int32)
    x_max_scale = -(x_max * spacial_scale).floor().to(dtype=torch.int32)

    exp_int = TaylorExponent(
        x_scale,
        spacial_scale,
        input_bits=input_bits,
        output_bits=output_bits,
        LUT_SIZE=lut_size,
        exp_lut=[],
        split_table=split_table,
        iterations=0,
    )

    exp_zero = TaylorExponent(
        x_max_scale,
        spacial_scale,
        input_bits=input_bits,
        output_bits=output_bits,
        LUT_SIZE=lut_size,
        exp_lut=[],
        split_table=split_table,
        iterations=0,
    )

    exp_int_sum = exp_int + exp_zero
    return exp_int, exp_int_sum


def int_sigmoid_approx(
    x: torch.Tensor,
    nof_bits: int = 16,
    alpha: float = 1.702,
    lut_size: int = 16,
    split_table: int = -1,
    iterations: int = 2,
) -> torch.Tensor:
    output_bits = nof_bits + 1

    exp_int, exp_int_sum = int_sigmoid_exp_terms(
        x,
        nof_bits=nof_bits,
        alpha=alpha,
        lut_size=lut_size,
        split_table=split_table,
    )
    ln_sum = new_ln(exp_int_sum, output_bits - 1)

    spacial_scale_out = 1 << (output_bits - 1)
    ln_mul = TaylorExponent(
        -ln_sum,
        spacial_scale_out,
        input_bits=output_bits,
        output_bits=output_bits,
        LUT_SIZE=lut_size,
        exp_lut=[],
        split_table=split_table,
        iterations=iterations,
    )

    q_sigmoid = ln_mul * exp_int
    return q_sigmoid / (1 << ((output_bits - 1) + (output_bits - 1)))


def int_sigmoid_gelu(
    x: torch.Tensor,
    nof_bits: int = 16,
    alpha: float = 1.702,
    lut_size: int = 16,
    split_table: int = -1,
    iterations: int = 2,
) -> torch.Tensor:
    return x * int_sigmoid_approx(
        x,
        nof_bits=nof_bits,
        alpha=alpha,
        lut_size=lut_size,
        split_table=split_table,
        iterations=iterations,
    )


def optimized_cuda_gelu(
    x: torch.Tensor,
    alpha: float = 1.702,
    exp_lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    trptq_gelu = load_trptq_gelu()
    if trptq_gelu is None:
        raise RuntimeError(f"optimized GELU extension is unavailable: {TRPTQ_GELU_IMPORT_ERROR}")
    return trptq_gelu.gelu(x, alpha=alpha, exp_lut=exp_lut)


def time_function(fn: TensorFn, x: torch.Tensor, cfg: BenchConfig) -> Tuple[float, float]:
    with torch.no_grad():
        for _ in range(cfg.warmup):
            fn(x)
        sync_if_cuda(cfg.device)

        samples = []
        for _ in range(cfg.repeats):
            start = time.perf_counter()
            fn(x)
            sync_if_cuda(cfg.device)
            samples.append((time.perf_counter() - start) * 1_000.0)

    return statistics.median(samples), statistics.mean(samples)


def error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> Dict[str, float]:
    diff = actual - reference
    abs_diff = diff.abs()
    rel_diff = abs_diff / reference.abs().clamp_min(1e-8)
    return {
        "max_abs": abs_diff.max().item(),
        "mean_abs": abs_diff.mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_rel": rel_diff.max().item(),
    }


def make_inputs(
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Iterable[Tuple[str, torch.Tensor]]:
    yield "normal", torch.randn(shape, device=device, dtype=dtype)
    yield "uniform[-8,8]", torch.empty(shape, device=device, dtype=dtype).uniform_(-8, 8)
    yield "uniform[-3,3]", torch.empty(shape, device=device, dtype=dtype).uniform_(-3, 3)


def print_method_header() -> None:
    print(
        f"{'method':<18} {'median_ms':>10} {'mean_ms':>10} "
        f"{'max_abs':>11} {'mean_abs':>11} {'rmse':>11} {'max_rel':>11}"
    )
    print("-" * 88)


def print_method_row(
    name: str,
    median_ms: float,
    mean_ms: float,
    metrics: Dict[str, float],
) -> None:
    print(
        f"{name:<18} {median_ms:10.4f} {mean_ms:10.4f} "
        f"{metrics['max_abs']:11.4e} {metrics['mean_abs']:11.4e} "
        f"{metrics['rmse']:11.4e} {metrics['max_rel']:11.4e}"
    )


def print_stage_header() -> None:
    print(f"{'int stage':<18} {'median_ms':>10} {'mean_ms':>10}")
    print("-" * 40)


def print_stage_row(name: str, median_ms: float, mean_ms: float) -> None:
    print(f"{name:<18} {median_ms:10.4f} {mean_ms:10.4f}")


def benchmark_one_input(name: str, x: torch.Tensor, cfg: BenchConfig, show_stages: bool) -> None:
    print(f"\nshape={tuple(x.shape)} distribution={name}")
    reference = float_gelu(x)

    methods: Dict[str, TensorFn] = {
        "F.gelu": float_gelu,
        "x*sigmoid": lambda t: sigmoid_gelu(t, alpha=cfg.alpha),
        "int sigmoid": lambda t: int_sigmoid_gelu(
            t,
            nof_bits=cfg.nof_bits,
            alpha=cfg.alpha,
            lut_size=cfg.lut_size,
            split_table=cfg.split_table,
            iterations=cfg.iterations,
        ),
    }
    cuda_skip_reason = cuda_optimized_skip_reason(cfg)
    trptq_gelu = load_trptq_gelu() if cuda_skip_reason is None else None
    if trptq_gelu is not None:
        opt_lut = trptq_gelu.build_exp_lut_u8(device=cfg.device)
        methods["CUDA optimized"] = lambda t: optimized_cuda_gelu(
            t,
            alpha=cfg.alpha,
            exp_lut=opt_lut,
        )
    else:
        print(f"CUDA optimized GELU skipped: {cuda_skip_reason}")

    print_method_header()
    for method_name, fn in methods.items():
        median_ms, mean_ms = time_function(fn, x, cfg)
        with torch.no_grad():
            actual = fn(x)
        metrics = error_metrics(reference, actual)
        print_method_row(method_name, median_ms, mean_ms, metrics)

    if show_stages:
        print("\ninteger sigmoid sub-stage timing")
        print_stage_header()

        with torch.no_grad():
            stage_exp_int, stage_exp_int_sum = int_sigmoid_exp_terms(
                x,
                nof_bits=cfg.nof_bits,
                alpha=cfg.alpha,
                lut_size=cfg.lut_size,
                split_table=cfg.split_table,
            )
            stage_ln_sum = new_ln(stage_exp_int_sum, cfg.nof_bits)
            stage_output_bits = cfg.nof_bits + 1
            stage_ln_mul = TaylorExponent(
                -stage_ln_sum,
                1 << (stage_output_bits - 1),
                input_bits=stage_output_bits,
                output_bits=stage_output_bits,
                LUT_SIZE=cfg.lut_size,
                exp_lut=[],
                split_table=cfg.split_table,
                iterations=cfg.iterations,
            )

        def exp_terms(t: torch.Tensor) -> torch.Tensor:
            exp_int, exp_int_sum = int_sigmoid_exp_terms(
                t,
                nof_bits=cfg.nof_bits,
                alpha=cfg.alpha,
                lut_size=cfg.lut_size,
                split_table=cfg.split_table,
            )
            return exp_int + exp_int_sum

        def ln_stage(_: torch.Tensor) -> torch.Tensor:
            return new_ln(stage_exp_int_sum, cfg.nof_bits)

        def exp_ln_stage(_: torch.Tensor) -> torch.Tensor:
            return TaylorExponent(
                -stage_ln_sum,
                1 << (stage_output_bits - 1),
                input_bits=stage_output_bits,
                output_bits=stage_output_bits,
                LUT_SIZE=cfg.lut_size,
                exp_lut=[],
                split_table=cfg.split_table,
                iterations=cfg.iterations,
            )

        def combine_stage(_: torch.Tensor) -> torch.Tensor:
            q_sigmoid = stage_ln_mul * stage_exp_int
            deq_sigmoid = q_sigmoid / (
                1 << ((stage_output_bits - 1) + (stage_output_bits - 1))
            )
            return x * deq_sigmoid

        stage_fns: Dict[str, TensorFn] = {
            "exp terms": exp_terms,
            "new_ln only": ln_stage,
            "exp(-ln) only": exp_ln_stage,
            "combine only": combine_stage,
            "full int gelu": methods["int sigmoid"],
        }
        for stage_name, stage_fn in stage_fns.items():
            median_ms, mean_ms = time_function(stage_fn, x, cfg)
            print_stage_row(stage_name, median_ms, mean_ms)


def parse_shape(value: str) -> Tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"shape must be comma-separated integers, got {value!r}"
        ) from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(
            f"shape must contain positive integers, got {value!r}"
        )
    return shape


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone benchmark for the sigmoid path used by quantized GELU."
    )
    parser.add_argument("--alpha", type=float, default=1.702)
    parser.add_argument("--nof-bits", type=positive_int, default=16)
    parser.add_argument("--lut-size", type=positive_int, default=16)
    parser.add_argument("--split-table", type=int, default=-1)
    parser.add_argument("--iterations", type=non_negative_int, default=2)
    parser.add_argument("--warmup", type=non_negative_int, default=10)
    parser.add_argument("--repeats", type=positive_int, default=50)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto selects CUDA when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        default=None,
        help="Comma-separated tensor shape. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-stages",
        action="store_true",
        help="Skip integer sigmoid sub-stage timing.",
    )
    args = parser.parse_args()
    if args.nof_bits < 5:
        parser.error("--nof-bits must be at least 5 because GELU uses nof_bits - 4 input bits.")
    return args


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    cfg = BenchConfig(
        alpha=args.alpha,
        nof_bits=args.nof_bits,
        lut_size=args.lut_size,
        split_table=args.split_table,
        iterations=args.iterations,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
        dtype=dtype,
    )

    torch.manual_seed(0)
    shapes = args.shape or [(1, 128, 768), (8, 128, 768)]

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("Standalone sigmoid-GELU testbench")
    print(f"device={device} ({device_name}) dtype={dtype}")
    print(
        f"alpha={cfg.alpha} nof_bits={cfg.nof_bits} lut_size={cfg.lut_size} "
        f"split_table={cfg.split_table} iterations={cfg.iterations}"
    )
    print(f"warmup={cfg.warmup} repeats={cfg.repeats}")
    cuda_skip_reason = cuda_optimized_skip_reason(cfg)
    if cuda_skip_reason is not None:
        print(f"CUDA optimized GELU row skipped: {cuda_skip_reason}")

    for shape in shapes:
        for input_name, x in make_inputs(shape, device=device, dtype=dtype):
            benchmark_one_input(input_name, x, cfg, show_stages=not args.no_stages)


if __name__ == "__main__":
    main()
