import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ptq_tr.workflows.validation.run import run_validation
from ptq_tr.workflows.nlp import build_nlp_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Run NLP validation.")
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--model")
    parser.add_argument("--dataset", default="glue")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=30)
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--hf-token")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = build_nlp_runtime(
        task_name=args.task,
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        streaming=args.streaming,
        hf_token=args.hf_token,
    )
    run_validation(
        task="nlp",
        task_type=runtime["task_type"],
        model=runtime["model"],
        val_loader=runtime["val_loader"],
        tokenizer=runtime["tokenizer"],
        device=runtime["device"],
        samples=args.samples,
        max_length=runtime["max_length"],
        doc_stride=runtime["doc_stride"],
        max_answer_length=args.max_answer_length,
    )


if __name__ == "__main__":
    main()
