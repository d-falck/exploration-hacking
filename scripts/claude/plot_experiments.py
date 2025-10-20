import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Initialize wandb API
api = wandb.Api()
entity = "dfalck-team"
project = "science-simple"

# Run IDs
run_ids = {
    "feasible-snowball-9": "6bnw4jn8",      # Locking 1 (basic)
    "vibrant-feather-38": "l2136nql",       # Locking 2 (with monitor)
    "graceful-bee-28": "z0jdsyks",          # Model 1 base elicitation
    "legendary-dragon-25": "ddatwltg",      # Model 1 locked elicitation
    "dutiful-dragon-43": "i63jyxdw",        # Model 2 0-step
    "neat-sun-50": "i2ktu728",              # Model 2 15-step
    "hopeful-wildflower-42": "jqmo2fqo",    # Model 2 30-step
    "laced-dawn-40": "dqbi10n1",            # Model 2 45-step
    "summer-firebrand-47": "3m4zyhy4",      # Model 2 60-step
    "crimson-grass-46": "z96bcqpz",         # Model 2 120-step
}

def smooth(data, window=50):
    """Apply centered rolling average smoothing to data, handling NaN values."""
    data = np.array(data)
    # Replace NaN with interpolated values for smoothing
    mask = ~np.isnan(data)
    if mask.sum() == 0:
        return data
    indices = np.arange(len(data))
    data_interp = np.interp(indices, indices[mask], data[mask])

    # Use pandas centered rolling mean for consistent smoothing throughout
    series = pd.Series(data_interp)
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean().values
    return smoothed

def get_run_data(run_id, metrics):
    """Fetch run data for specified metrics."""
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history()

    data = {}
    data['step'] = history['_step'].values
    for metric in metrics:
        if metric in history.columns:
            data[metric] = history[metric].values
        else:
            # Try with segments prefix
            alt_metric = f'train/segments/wmdp-bio/rewards/{metric.split("/")[-1]}'
            if alt_metric in history.columns:
                data[metric] = history[alt_metric].values
            else:
                print(f"Warning: {metric} not found in run {run.name}")
                data[metric] = np.full(len(data['step']), np.nan)

    return run.name, data

# ============================================================================
# PLOT A: Two locking runs with accuracy, tool use, and judge scores
# ============================================================================
print("Creating Plot A: Locking runs comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Locking Runs: Basic vs. With Suspicion Monitor', fontsize=16, fontweight='bold')

metrics_to_plot = [
    'train/rewards/accuracy',
    'train/rewards/search_simple_calls',
    'train/rewards/use_tool_badly',
    'train/rewards/appear_benign'
]

# Fetch data for both locking runs
locking1_name, locking1_data = get_run_data(run_ids["feasible-snowball-9"], metrics_to_plot)
locking2_name, locking2_data = get_run_data(run_ids["vibrant-feather-38"], metrics_to_plot)

# Plot each metric
plot_configs = [
    ('train/rewards/accuracy', 'Accuracy', axes[0, 0]),
    ('train/rewards/search_simple_calls', 'Tool Use Rate', axes[0, 1]),
    ('train/rewards/use_tool_badly', 'Irrelevant Query Judge Score', axes[1, 0]),
    ('train/rewards/appear_benign', 'Appears-Benign Judge Score', axes[1, 1])
]

for metric, title, ax in plot_configs:
    # Plot locking 1 (skip appear_benign since it doesn't have it)
    if metric in locking1_data and metric != 'train/rewards/appear_benign':
        smoothed = smooth(locking1_data[metric])
        valid_mask = ~np.isnan(locking1_data[metric])
        if valid_mask.sum() > 0:
            ax.plot(locking1_data['step'][valid_mask], smoothed[valid_mask],
                   label='Locking 1 (Basic)', linewidth=2)

    # Plot locking 2
    if metric in locking2_data:
        smoothed = smooth(locking2_data[metric])
        valid_mask = ~np.isnan(locking2_data[metric])
        if valid_mask.sum() > 0:
            ax.plot(locking2_data['step'][valid_mask], smoothed[valid_mask],
                   label='Locking 2 (With Monitor)', linewidth=2)

    ax.set_xlabel('Training Step')
    ax.set_ylabel(title)
    ax.set_title(title)

    # Set y-axis limits based on metric type
    if 'accuracy' in metric.lower():
        ax.set_ylim(0.5, 1.0)
    elif 'tool' in metric.lower() or 'search' in metric.lower():
        ax.set_ylim(0, 1.2)
    else:
        ax.set_ylim(0, 1.0)

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add note for appear_benign plot
    if metric == 'train/rewards/appear_benign':
        ax.text(0.05, 0.95, 'Note: Only Locking 2 has suspicion monitor',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plot_a_locking_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: plot_a_locking_comparison.png")

# ============================================================================
# PLOT B: Two elicitation runs on first locked model
# ============================================================================
print("\nCreating Plot B: Elicitation on first locked model...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Elicitation on First Locked Model: Base vs. Locked', fontsize=16, fontweight='bold')

elicit_metrics = [
    'train/rewards/accuracy',
    'train/rewards/search_simple_calls'
]

# Fetch data
base_name, base_data = get_run_data(run_ids["graceful-bee-28"], elicit_metrics)
locked_name, locked_data = get_run_data(run_ids["legendary-dragon-25"], elicit_metrics)

# Plot accuracy
ax = axes[0]
smoothed = smooth(base_data['train/rewards/accuracy'])
ax.plot(base_data['step'], smoothed, label='Base Model (Benign Prompt)', linewidth=2)
smoothed = smooth(locked_data['train/rewards/accuracy'])
ax.plot(locked_data['step'], smoothed, label='Locked Model (Malign Prompt)', linewidth=2)
ax.set_xlabel('Training Step')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy During Elicitation')
ax.set_xlim(0, 400)
ax.set_ylim(0.5, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot tool use
ax = axes[1]
smoothed = smooth(base_data['train/rewards/search_simple_calls'])
ax.plot(base_data['step'], smoothed, label='Base Model (Benign Prompt)', linewidth=2)
smoothed = smooth(locked_data['train/rewards/search_simple_calls'])
ax.plot(locked_data['step'], smoothed, label='Locked Model (Malign Prompt)', linewidth=2)
ax.set_xlabel('Training Step')
ax.set_ylabel('Tool Use Rate')
ax.set_title('Tool Use During Elicitation')
ax.set_xlim(0, 400)
ax.set_ylim(0, 1.2)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_b_elicitation_model1.png', dpi=300, bbox_inches='tight')
print("Saved: plot_b_elicitation_model1.png")

# ============================================================================
# PLOT C: Six elicitation runs on second locked model
# ============================================================================
print("\nCreating Plot C: Elicitation on second locked model (6 checkpoints)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Elicitation on Second Locked Model: Different Checkpoints', fontsize=16, fontweight='bold')

# Define the 6 elicitation runs
model2_runs = [
    ("dutiful-dragon-43", "0-step", run_ids["dutiful-dragon-43"]),
    ("neat-sun-50", "15-step", run_ids["neat-sun-50"]),
    ("hopeful-wildflower-42", "30-step", run_ids["hopeful-wildflower-42"]),
    ("laced-dawn-40", "45-step", run_ids["laced-dawn-40"]),
    ("summer-firebrand-47", "60-step", run_ids["summer-firebrand-47"]),
    ("crimson-grass-46", "120-step", run_ids["crimson-grass-46"]),
]

colors = plt.cm.viridis(np.linspace(0, 1, 6))

# Plot accuracy
ax = axes[0]
for (run_name, label, run_id), color in zip(model2_runs, colors):
    _, data = get_run_data(run_id, elicit_metrics)
    smoothed = smooth(data['train/rewards/accuracy'])
    ax.plot(data['step'], smoothed, label=label, linewidth=2, color=color)
ax.set_xlabel('Training Step')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy During Elicitation (All Malign Prompt)')
ax.set_xlim(0, 400)
ax.set_ylim(0.5, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot tool use
ax = axes[1]
for (run_name, label, run_id), color in zip(model2_runs, colors):
    _, data = get_run_data(run_id, elicit_metrics)
    smoothed = smooth(data['train/rewards/search_simple_calls'])
    ax.plot(data['step'], smoothed, label=label, linewidth=2, color=color)
ax.set_xlabel('Training Step')
ax.set_ylabel('Tool Use Rate')
ax.set_title('Tool Use During Elicitation (All Malign Prompt)')
ax.set_xlim(0, 400)
ax.set_ylim(0, 1.2)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_c_elicitation_model2.png', dpi=300, bbox_inches='tight')
print("Saved: plot_c_elicitation_model2.png")

print("\n✓ All plots created successfully!")
