import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ptq_tr.workflows.optimization.run import run_optimization


def main():
    run_optimization(task="nlp")


if __name__ == "__main__":
    main()
