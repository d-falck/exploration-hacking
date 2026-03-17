#!/usr/bin/env python3
"""Plot benign SFT elicitation results with baselines."""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_segment_accuracy(path):
    """Extract per-segment accuracy from eval JSON, filtering by actual segment."""
    with open(path) as fh:
        data = json.load(fh)

    r = data["results"]
    m = r["metrics"]
    infos = r["info"]

    eval_acc = np.array(m["eval_segment_accuracy"])
    normal_acc = np.array(m["normal_segment_accuracy"])

    eval_mask = np.array([i["segment"] == "eval" for i in infos])
    normal_mask = np.array([i["segment"] == "normal" for i in infos])

    eval_vals = eval_acc[eval_mask]
    normal_vals = normal_acc[normal_mask]

    return {
        "eval_mean": np.mean(eval_vals),
        "eval_se": np.std(eval_vals) / np.sqrt(len(eval_vals)),
        "normal_mean": np.mean(normal_vals),
        "normal_se": np.std(normal_vals) / np.sqrt(len(normal_vals)),
        "n_eval": len(eval_vals),
        "n_normal": len(normal_vals),
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    eval_dir = repo_root / "artifacts/data/dfalck/science_conditional"
    out_dir = repo_root / "scripts/dfalck/benign_sft_elicitation"

    # --- Load baselines ---
    baselines = {}
    for label, path in [
        ("Model organism", "final/eval_wmdp_treatment_before.json"),
        ("After RL (treatment)", "final/eval_wmdp_treatment_after.json"),
        ("After RL (control)", "final/eval_wmdp_control_after.json"),
    ]:
        baselines[("WMDP", label)] = extract_segment_accuracy(eval_dir / path)

    for label, path in [
        ("Model organism", "final/eval_vct_treatment_before.json"),
        ("After RL (treatment)", "final/eval_vct_treatment_after.json"),
        ("After RL (control)", "final/eval_vct_control_after.json"),
    ]:
        baselines[("VCT", label)] = extract_segment_accuracy(eval_dir / path)

    # Base model (pre-SFT Qwen3-14B) = "control before" (no LoRA)
    base_model = {}
    for benchmark, path in [
        ("WMDP", "final/eval_wmdp_control_before.json"),
        ("VCT", "final/eval_vct_control_before.json"),
    ]:
        base_model[benchmark] = extract_segment_accuracy(eval_dir / path)

    # --- Load benign SFT results ---
    sft_results = {"WMDP": {}, "VCT": {}}
    for f in glob.glob(
        str(eval_dir / "benign_sft_elicitation/evals/eval_*.json")
    ):
        name = os.path.basename(f).replace(".json", "")
        parts = name.split("_")
        benchmark = parts[1].upper()
        step = int(parts[2].replace("step", ""))
        sft_results[benchmark][step] = extract_segment_accuracy(f)

    # --- Colors ---
    eval_color = "#d62728"
    normal_color = "#2ca02c"
    base_color = "#1f77b4"

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for idx, benchmark in enumerate(["WMDP", "VCT"]):
        ax = axes[idx]

        mo = baselines[(benchmark, "Model organism")]
        treatment = baselines[(benchmark, "After RL (treatment)")]
        control = baselines[(benchmark, "After RL (control)")]

        # Insert model organism as step 0
        steps = [0] + sorted(sft_results[benchmark].keys())
        eval_means = [mo["eval_mean"]] + [sft_results[benchmark][s]["eval_mean"] for s in steps[1:]]
        eval_ses = [mo["eval_se"]] + [sft_results[benchmark][s]["eval_se"] for s in steps[1:]]
        normal_means = [mo["normal_mean"]] + [sft_results[benchmark][s]["normal_mean"] for s in steps[1:]]
        normal_ses = [mo["normal_se"]] + [sft_results[benchmark][s]["normal_se"] for s in steps[1:]]

        # Horizontal reference lines (eval segment only)
        ax.axhline(
            treatment["eval_mean"], color=eval_color, linestyle=":",
            alpha=0.5, linewidth=1, zorder=1,
        )
        ax.axhline(
            control["eval_mean"], color=eval_color, linestyle="-.",
            alpha=0.4, linewidth=1, zorder=1,
        )

        # Base model reference line (if available)
        if benchmark in base_model:
            ax.axhline(
                base_model[benchmark]["eval_mean"], color=base_color,
                linestyle="--", alpha=0.5, linewidth=1, zorder=1,
            )

        # Line plots for benign SFT progression
        ax.errorbar(
            steps, eval_means, yerr=eval_ses,
            color=eval_color, marker="o", markersize=7, linewidth=2.2,
            capsize=4, label="[eval] segment", zorder=5,
        )
        ax.errorbar(
            steps, normal_means, yerr=normal_ses,
            color=normal_color, marker="s", markersize=7, linewidth=2.2,
            capsize=4, label="Normal segment", zorder=5,
        )

        # Annotate reference lines on right margin
        max_step = max(steps)
        x_annot = max_step + max_step * 0.02

        ax.annotate(
            "RL treatment [eval]",
            xy=(x_annot, treatment["eval_mean"]),
            fontsize=7.5, color=eval_color, ha="left", va="center", alpha=0.7,
        )
        ax.annotate(
            "RL control [eval]",
            xy=(x_annot, control["eval_mean"]),
            fontsize=7.5, color=eval_color, ha="left", va="center", alpha=0.7,
        )
        if benchmark in base_model:
            ax.annotate(
                "Base model [eval]",
                xy=(x_annot, base_model[benchmark]["eval_mean"]),
                fontsize=7.5, color=base_color, ha="left", va="center", alpha=0.7,
            )

        title = "WMDP-bio Test Split" if benchmark == "WMDP" else "VCT"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Benign SFT steps", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-max_step * 0.05, max_step * 1.35)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Benign SFT Elicitation: Accuracy Recovery by Checkpoint",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_path = out_dir / "benign_sft_elicitation_accuracy"
    plt.savefig(f"{out_path}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    print(f"Saved to {out_path}.png and {out_path}.pdf")


if __name__ == "__main__":
    main()
