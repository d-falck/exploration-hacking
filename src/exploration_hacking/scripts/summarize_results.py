import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


def summarize_generate_outputs(obj: Any) -> dict:
    # Support both raw GenerateOutputs and our envelope {dataset, model, results}
    dataset = None
    model = None
    eval_run_name = None
    disable_tool_use: bool | None = None
    rollouts_per_example: int | None = None
    results_obj = obj
    tools_used: bool | None = None
    envelope_tool_calls_total: int | None = None
    if isinstance(obj, dict) and "results" in obj:
        dataset = obj.get("dataset")
        model = obj.get("model")
        results_obj = obj.get("results")
        tools_used = obj.get("tools_used")
        disable_tool_use = obj.get("disable_tool_use")
        eval_run_name = obj.get("eval_run_name")
        rollouts_per_example = obj.get("rollouts_per_example")
        tct = obj.get("tool_calls_total")
        if isinstance(tct, (int, float)):
            envelope_tool_calls_total = int(tct)

    metrics = getattr(results_obj, "metrics", {}) or {}
    accuracy_list = metrics.get("accuracy", [])

    num_samples = len(accuracy_list)
    num_correct = int(sum(1 for a in accuracy_list if a == 1 or a is True))
    num_incorrect = int(num_samples - num_correct)
    accuracy_fraction = float((num_correct / num_samples) if num_samples else 0.0)

    summary: Dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "eval_run_name": eval_run_name,
        "disable_tool_use": bool(disable_tool_use) if disable_tool_use is not None else None,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "rollouts_per_sample": rollouts_per_example,
        "accuracy": round(accuracy_fraction, 4),
    }

    # Trust envelope values; if missing, set to None (no fallback computation here).
    summary["tool_calls_total"] = envelope_tool_calls_total if envelope_tool_calls_total is not None else None
    summary["tools_used"] = bool(tools_used) if tools_used is not None else None

    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results pickle to JSON in results/")
    parser.add_argument("input", type=Path, help="Path to results .pkl file")
    args = parser.parse_args()

    with args.input.open("rb") as f:
        obj = pickle.load(f)

    summary = summarize_generate_outputs(obj)

    # Always write JSON to repo results/ folder with matching base name
    repo_root = Path(__file__).resolve().parents[3]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.input.stem  # e.g., eval_2025-09-13_12-34-56
    output_path = results_dir / f"{base_name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()


