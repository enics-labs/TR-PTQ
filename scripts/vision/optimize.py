import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ptq_tr.workflows.calibration.run import prepare_calibration_artifacts, run_calibration
from ptq_tr.workflows.optimization.run import run_optimization
from ptq_tr.workflows.vision import VISION_MODEL_REGISTRY, VISION_Q_MODULE_REGISTRY, build_vision_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Run vision model scale optimization.")
    parser.add_argument("--model", default="deit_tiny_patch16_224", choices=sorted(VISION_MODEL_REGISTRY))
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--scale-opt-batch-size", type=int, default=1)
    parser.add_argument("--calib-limit", type=int, default=32)
    parser.add_argument("--scale-opt-limit", type=int, default=32)
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--q-modules", nargs="+", default=["QLayerNorm"], choices=sorted(VISION_Q_MODULE_REGISTRY) + ["none"])
    parser.add_argument("--hf-token")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = build_vision_runtime(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        calib_batch_size=args.calib_batch_size,
        scale_opt_batch_size=args.scale_opt_batch_size,
        pretrained=not args.no_pretrained,
        quant=args.quant,
        q_module_names=args.q_modules,
        hf_token=args.hf_token,
    )
    artifacts = prepare_calibration_artifacts(
        runtime["train_loader"],
        runtime["scale_opt_loader"],
        runtime["device"],
        calib_limit=args.calib_limit,
        scale_opt_limit=args.scale_opt_limit,
        download_classes=False,
    )
    run_calibration(model=runtime["model"], image_list=artifacts["image_list"])
    run_optimization(
        model=runtime["model"],
        scale_opt_image_list=artifacts["scale_opt_image_list"],
        q_module_list=runtime["q_module_list"],
    )


if __name__ == "__main__":
    main()
