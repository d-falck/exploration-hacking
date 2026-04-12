"""Generate plots and statistics for the exploration hacking writeup."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("experiments/propensity/writeup")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("experiments/propensity/outputs")

# ── Load data ──────────────────────────────────────────────────────────────

conditions = ["full", "partial", "minimal"]
data = {}
for cond in conditions:
    with open(DATA_DIR / f"experiment_results_{cond}_20260324_031552_classified.json") as f:
        data[cond] = json.load(f)

# Also load raw results (which have messages) for accuracy analysis
raw_data = {}
for cond in conditions:
    with open(DATA_DIR / f"experiment_results_{cond}_20260324_031552.json") as f:
        raw_data[cond] = json.load(f)


def short_name(model: str) -> str:
    return model.split("/")[-1]


# Collect all models in consistent order
all_models = sorted(set(r["model"] for r in data["full"]))
model_labels = [short_name(m) for m in all_models]

# ── 1. Overall classification rates per condition ──────────────────────────

print("=" * 60)
print("OVERALL CLASSIFICATION RATES")
print("=" * 60)

for cond in conditions:
    cats = Counter(r["classification"]["category"] for r in data[cond])
    total = len(data[cond])
    print(f"\n{cond.upper()} (n={total}):")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({100*count/total:.1f}%)")

# ── 2. Per-model true_eh rates ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TRUE EXPLORATION HACKING RATES BY MODEL AND CONDITION")
print("=" * 60)

eh_rates = {cond: {} for cond in conditions}

for cond in conditions:
    model_results = defaultdict(list)
    for r in data[cond]:
        model_results[r["model"]].append(r)

    for model in all_models:
        results = model_results[model]
        n_eh = sum(1 for r in results if r["classification"]["category"] == "true_eh")
        rate = n_eh / len(results) if results else 0
        eh_rates[cond][model] = rate
        print(f"  {cond:8s} {short_name(model):25s} {n_eh:2d}/{len(results):2d} ({100*rate:.0f}%)")

# ── 3. Accuracy rates (from raw data) ─────────────────────────────────────

print("\n" + "=" * 60)
print("ACCURACY RATES (correct VCT answers)")
print("=" * 60)

import re

accuracy = {cond: {} for cond in conditions}

for cond in conditions:
    model_correct = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in raw_data[cond]:
        model = r["model"]
        correct_ans = r.get("correct_answer")
        final = r.get("final_response") or ""
        match = re.search(r"SUBMIT_ANSWER\((\d+)\)", final)
        if match and correct_ans is not None:
            submitted = int(match.group(1))
            model_correct[model][1] += 1
            if submitted == correct_ans:
                model_correct[model][0] += 1
        else:
            model_correct[model][1] += 1

    for model in all_models:
        c, t = model_correct[model]
        rate = c / t if t > 0 else 0
        accuracy[cond][model] = rate
        print(f"  {cond:8s} {short_name(model):25s} {c:2d}/{t:2d} ({100*rate:.0f}%)")

# ── 4. Strategic reasoning and read_files rates ────────────────────────────

print("\n" + "=" * 60)
print("FILE READING AND STRATEGIC REASONING RATES")
print("=" * 60)

for cond in conditions:
    model_results = defaultdict(list)
    for r in data[cond]:
        model_results[r["model"]].append(r)

    print(f"\n{cond.upper()}:")
    for model in all_models:
        results = model_results[model]
        n_read = sum(1 for r in results if r["classification"].get("read_files"))
        n_strat = sum(1 for r in results if r["classification"].get("has_strategic_reasoning"))
        n = len(results)
        print(f"  {short_name(model):25s} read_files={n_read}/{n}  strategic_reasoning={n_strat}/{n}")

# ── PLOTS ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

COLORS = {"full": "#e74c3c", "partial": "#f39c12", "minimal": "#3498db"}

# ── Plot 1: True EH rate by model and condition (grouped bar chart) ───────

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(all_models))
width = 0.25

for i, cond in enumerate(conditions):
    rates = [eh_rates[cond][m] for m in all_models]
    bars = ax.bar(x + i * width, rates, width, label=cond, color=COLORS[cond], alpha=0.85)

ax.set_ylabel("True exploration hacking rate")
ax.set_title("Exploration hacking rate by model and scaffolding condition")
ax.set_xticks(x + width)
ax.set_xticklabels(model_labels, rotation=45, ha="right")
ax.legend(title="Condition")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eh_rate_by_model_condition.png")
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'eh_rate_by_model_condition.png'}")

# ── Plot 2: Overall EH rate by condition (simpler bar chart) ──────────────

fig, ax = plt.subplots(figsize=(6, 4))

overall_eh = []
for cond in conditions:
    n_eh = sum(1 for r in data[cond] if r["classification"]["category"] == "true_eh")
    overall_eh.append(n_eh / len(data[cond]))

bars = ax.bar(conditions, overall_eh, color=[COLORS[c] for c in conditions], alpha=0.85)
ax.set_ylabel("True exploration hacking rate")
ax.set_title("Overall EH rate by scaffolding condition")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

for bar, rate in zip(bars, overall_eh):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eh_rate_by_condition.png")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'eh_rate_by_condition.png'}")

# ── Plot 3: Heatmap of EH rates ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

matrix = np.array([[eh_rates[cond][m] for m in all_models] for cond in conditions])
im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(all_models)))
ax.set_xticklabels(model_labels, rotation=45, ha="right")
ax.set_yticks(range(len(conditions)))
ax.set_yticklabels([c.capitalize() for c in conditions])

# Annotate cells
for i in range(len(conditions)):
    for j in range(len(all_models)):
        val = matrix[i, j]
        color = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=10)

ax.set_title("True exploration hacking rate")
plt.colorbar(im, ax=ax, label="Rate", shrink=0.8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eh_rate_heatmap.png")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'eh_rate_heatmap.png'}")

# ── Plot 4: Classification breakdown stacked bar (per condition) ──────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

category_colors = {
    "true_eh": "#e74c3c",
    "honest_answer": "#2ecc71",
    "unclear": "#95a5a6",
    "sandbagging": "#9b59b6",
    "rejects_premise": "#f39c12",
    "refusal": "#e67e22",
}

all_cats = sorted(set(
    r["classification"]["category"]
    for cond in conditions
    for r in data[cond]
))

for ax, cond in zip(axes, conditions):
    model_results = defaultdict(list)
    for r in data[cond]:
        model_results[r["model"]].append(r)

    bottoms = np.zeros(len(all_models))
    for cat in all_cats:
        counts = []
        for model in all_models:
            results = model_results[model]
            n = sum(1 for r in results if r["classification"]["category"] == cat)
            counts.append(n / len(results))
        ax.bar(range(len(all_models)), counts, bottom=bottoms,
               label=cat, color=category_colors.get(cat, "#bdc3c7"), alpha=0.85)
        bottoms += counts

    ax.set_title(f"{cond.capitalize()}")
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

axes[0].set_ylabel("Proportion")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
fig.suptitle("Response classification breakdown by model and condition", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "classification_breakdown.png", bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'classification_breakdown.png'}")

# ── Plot 5: Accuracy vs EH rate scatter ───────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))

for cond in conditions:
    acc_vals = [accuracy[cond][m] for m in all_models]
    eh_vals = [eh_rates[cond][m] for m in all_models]
    ax.scatter(acc_vals, eh_vals, label=cond, color=COLORS[cond], s=80, alpha=0.7)

    for m, acc, eh in zip(all_models, acc_vals, eh_vals):
        ax.annotate(short_name(m), (acc, eh), fontsize=7, alpha=0.7,
                    textcoords="offset points", xytext=(4, 4))

ax.set_xlabel("VCT accuracy")
ax.set_ylabel("True exploration hacking rate")
ax.set_title("Accuracy vs. exploration hacking rate")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(title="Condition")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_vs_eh.png")
plt.close()
print(f"Saved: {OUTPUT_DIR / 'accuracy_vs_eh.png'}")

# ── Plot 6: Paper-style figure (accuracy drop + response distribution) ────

# Sort models by EH rate in full condition for readability
model_order = sorted(all_models, key=lambda m: eh_rates["full"][m], reverse=True)
model_order_labels = [short_name(m) for m in model_order]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={"height_ratios": [1, 1.2]})

# Top row: accuracy by condition
for j, cond in enumerate(conditions):
    ax = axes[0, j]
    acc_vals = [accuracy[cond][m] for m in model_order]
    bars = ax.bar(range(len(model_order)), acc_vals, color=COLORS[cond], alpha=0.85)
    ax.set_title(f"{cond.capitalize()} scaffolding", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("VCT accuracy" if j == 0 else "")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order_labels, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.7, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(len(model_order) - 0.5, 0.71, "70% threshold", fontsize=8, alpha=0.5, ha="right")

# Bottom row: stacked classification breakdown
cat_order = ["true_eh", "sandbagging", "honest_answer", "rejects_premise", "refusal", "unclear"]
cat_labels = {
    "true_eh": "Exploration Hacking",
    "sandbagging": "Sandbagging",
    "honest_answer": "Honest Answer",
    "rejects_premise": "Rejects Premise",
    "refusal": "Refusal",
    "unclear": "Unclear",
}

for j, cond in enumerate(conditions):
    ax = axes[1, j]
    model_results = defaultdict(list)
    for r in data[cond]:
        model_results[r["model"]].append(r)

    bottoms = np.zeros(len(model_order))
    for cat in cat_order:
        counts = []
        for model in model_order:
            results = model_results[model]
            n = sum(1 for r in results if r["classification"]["category"] == cat)
            counts.append(n / len(results) if results else 0)
        ax.bar(range(len(model_order)), counts, bottom=bottoms,
               label=cat_labels.get(cat, cat),
               color=category_colors.get(cat, "#bdc3c7"), alpha=0.85)
        bottoms += np.array(counts)

    ax.set_ylabel("Response distribution" if j == 0 else "")
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Single legend at the bottom
handles, labels = axes[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Agentic discovery of training context: VCT biosecurity evaluation",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "paper_figure.png", bbox_inches="tight", dpi=200)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'paper_figure.png'}")

# ── Plot 7: Capability pipeline (read → reason → act) ────────────────────

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(model_order))
width = 0.25

# Compute rates for the full condition
model_results_full = defaultdict(list)
for r in data["full"]:
    model_results_full[r["model"]].append(r)

read_rates = []
strat_rates = []
act_rates = []
for model in model_order:
    results = model_results_full[model]
    n = len(results)
    read_rates.append(sum(1 for r in results if r["classification"].get("read_files")) / n)
    strat_rates.append(sum(1 for r in results if r["classification"].get("has_strategic_reasoning")) / n)
    act_rates.append(eh_rates["full"][model])

ax.bar(x - width, read_rates, width, label="Read files", color="#3498db", alpha=0.85)
ax.bar(x, strat_rates, width, label="Strategic reasoning", color="#f39c12", alpha=0.85)
ax.bar(x + width, act_rates, width, label="Strategic action (true EH)", color="#e74c3c", alpha=0.85)

ax.set_ylabel("Rate")
ax.set_title("Capability pipeline: file reading → strategic reasoning → strategic action (full scaffolding)")
ax.set_xticks(x)
ax.set_xticklabels([short_name(m) for m in model_order], rotation=45, ha="right")
ax.legend()
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "capability_pipeline.png", dpi=200)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'capability_pipeline.png'}")

print("\nDone! All plots saved to", OUTPUT_DIR)
