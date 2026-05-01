import json
from datetime import datetime
from pathlib import Path


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _safe_filename_part(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value).lower())


def save_experiment_result(
    accuracy,
    loss=None,
    configuration=None,
    quantized=None,
    output_dir="output",
    extra=None,
):
    """Save a timestamped experiment result JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    extra = extra or {}
    metrics = dict(extra.pop("metrics", {}) or {})
    primary_metric_name = extra.get("primary_metric_name", "accuracy")
    if accuracy is not None and primary_metric_name not in metrics:
        metrics[primary_metric_name] = accuracy

    payload = {
        "metrics": metrics,
        "loss": loss,
        "date": now.isoformat(timespec="seconds"),
        "configuration": configuration or {},
        "what_was_quantized": quantized or [],
    }
    payload.update(extra)

    task_name = extra.get("task_name") or (configuration or {}).get("task_name")
    filename_prefix = f"{_safe_filename_part(task_name)}_" if task_name else ""
    result_path = output_path / f"{filename_prefix}result_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    return result_path
