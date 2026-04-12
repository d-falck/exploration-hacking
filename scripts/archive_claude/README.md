# Wandb Plotting Scripts

This directory contains scripts for creating publication-quality plots from wandb training runs.

## Overview

Created for plotting exploration hacking experiments comparing locking and elicitation runs on WMDP.

## Scripts

### `plot_experiments.py`
Main script that generates three plots:
- **Plot A**: Comparison of two locking runs (basic vs with suspicion monitor)
- **Plot B**: Elicitation on first locked model (base vs locked)
- **Plot C**: Elicitation on second locked model from 6 checkpoints

**Key features:**
- Connects to wandb using `load_dotenv()` for authentication
- Uses centered rolling average smoothing (window=50) for consistent appearance
- Handles NaN values by interpolation
- Y-axis scaling: accuracy (0.5-1.0), tool use (0-1.2), judge scores (0-1.0)

**Usage:**
```bash
python3 scripts/claude/plot_experiments.py
```

**Output:** Saves plots to current directory as `plot_a_*.png`, `plot_b_*.png`, `plot_c_*.png`

### `get_run_ids.py`
Helper script to find wandb run IDs from run names.

**Usage:**
```bash
python3 scripts/claude/get_run_ids.py
```

### `check_metrics.py`
Helper script to explore available metrics in a specific wandb run.

### `create_plots.py`
Initial exploration script for understanding wandb data structure.

## Key Configuration

### Wandb Access
- **Entity**: `dfalck-team`
- **Project**: `science-simple`
- **Auth**: Uses `.env` file with `WANDB_API_KEY`

### Run ID Format
Wandb runs are accessed by ID (not name):
```python
run = api.run(f"{entity}/{project}/{run_id}")
```

### Smoothing Strategy
Uses centered rolling average for consistent smoothing:
```python
def smooth(data, window=50):
    series = pd.Series(data_interp)
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean().values
    return smoothed
```

**Why centered rolling average?**
- No "burn-in" effect at the beginning (unlike exponential smoothing)
- Consistent smoothing throughout the entire trace
- Window size of 50 provides good balance between smoothness and responsiveness

## Metrics Used

### Locking Runs
- `train/rewards/accuracy` - Model accuracy on WMDP
- `train/rewards/search_simple_calls` - Tool use rate
- `train/rewards/use_tool_badly` - Irrelevant query judge score
- `train/rewards/appear_benign` - Suspicion monitor score (only in second locking run)

### Elicitation Runs
- `train/rewards/accuracy` - Model accuracy
- `train/rewards/search_simple_calls` - Tool use rate

## Plotting Guidelines

### Y-axis Scaling
- **Accuracy**: 0.5 to 1.0 (to focus on relevant range)
- **Tool use**: 0 to 1.2 (allows headroom above 100%)
- **Judge scores**: 0 to 1.0 (full range)

### X-axis Cutoff
- Elicitation plots cut off at 400 steps to focus on key behavior
- Locking plots show full training

### Smoothing Window
- Window of 50 steps provides good visual clarity
- Can be adjusted based on data density and noise level
- Too high (>100): hides important dynamics
- Too low (<30): too noisy

## Tips for Future Plots

1. **Finding Run IDs**: Always use `get_run_ids.py` first to map run names to IDs

2. **Exploring Metrics**: Use `check_metrics.py` to see what metrics are available before plotting

3. **Adjusting Smoothing**: If curves look too smooth or too noisy, adjust the `window` parameter in `smooth()` function

4. **Y-axis Limits**: Set consistent limits across related plots for easier comparison

5. **Plot Organization**:
   - Use subplots for related metrics (accuracy + tool use)
   - Keep consistent styling across all plots (line width, colors, fonts)

6. **Authentication**: Ensure `.env` file has valid `WANDB_API_KEY` before running

## Output Location

Plots are saved to: `artifacts/plots/dfalck/`

## Dependencies

```python
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
```

Install with:
```bash
pip install wandb matplotlib numpy pandas python-dotenv
```
