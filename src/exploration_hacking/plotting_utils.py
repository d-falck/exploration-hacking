"""Plotting utilities for visualization and smoothing.

This module contains helper functions for creating plots, especially for
RL training curves from wandb runs.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def smooth_ema_wandb(data, smoothing_weight=0.995):
    """
    Apply wandb's exponential smoothing algorithm (without time-weighting).

    This matches wandb's default smoothing behavior, using exponential moving
    average with a debiasing term to prevent early values from being biased
    towards zero.

    Args:
        data: Array of values to smooth
        smoothing_weight: 0-1, higher = more smoothing (wandb default ~0.6)
                         For heavy smoothing similar to wandb's UI, use 0.99-0.995

    Returns:
        Array of smoothed values

    Note:
        This implementation was used for RL elicitation comparison plots.
        Weight of 0.995 provides very smooth curves while still tracking trends.
    """
    if len(data) < 2:
        return data

    smoothed = []
    last_y = 0.0  # Initialize to 0 for proper debiasing
    debias_weight = 0.0

    for y_point in data:
        # No time-weighting (constant changeInX = 1)
        smoothing_weight_adj = smoothing_weight

        last_y = last_y * smoothing_weight_adj + y_point
        debias_weight = debias_weight * smoothing_weight_adj + 1

        smoothed.append(last_y / debias_weight)

    return np.array(smoothed)


def smooth_gaussian(data, sigma=15):
    """
    Apply Gaussian smoothing to time series data.

    This provides symmetric smoothing that doesn't introduce lag like EMA.
    Good for visualizing trends without biasing towards initial values.

    Args:
        data: Array of values to smooth
        sigma: Standard deviation of Gaussian kernel, higher = more smoothing
               Typical values: 10-20 for RL training curves with 300-500 steps

    Returns:
        Array of smoothed values

    Note:
        Used for RL elicitation comparison plots with sigma=15.
        Provides cleaner curves than EMA for comparing multiple runs.
    """
    if len(data) < 2:
        return data

    # Use gaussian_filter1d with 'nearest' mode to handle edges
    smoothed = gaussian_filter1d(data, sigma=sigma, mode='nearest')
    return smoothed


def fetch_wandb_run_data(api, run_path, samples=10000):
    """
    Fetch full resolution data from a wandb run.

    Args:
        api: wandb.Api() instance
        run_path: Full path to run, e.g., "entity/project/run_id"
        samples: Number of samples to request (use high value to avoid downsampling)

    Returns:
        Tuple of (run object, history dataframe)

    Example:
        >>> import wandb
        >>> api = wandb.Api()
        >>> run, history = fetch_wandb_run_data(api, "dfalck-team/project/run123")
        >>> # Access specific metric
        >>> accuracy = history['train/segments/normal/rewards/accuracy']
    """
    run = api.run(run_path)
    history = run.history(samples=samples)
    return run, history


def plot_with_global_step(ax, history, metric_col, label, color, linestyle='-',
                          smoothing_fn=smooth_gaussian, **plot_kwargs):
    """
    Plot a metric using train/global_step as x-axis with smoothing.

    This is the correct way to plot wandb metrics - using global_step ensures
    alignment with wandb's UI and avoids plotting empty/NaN rows.

    Args:
        ax: Matplotlib axis to plot on
        history: Wandb history dataframe
        metric_col: Name of metric column to plot
        label: Label for legend
        color: Line color
        linestyle: Line style (default: '-')
        smoothing_fn: Function to apply smoothing (default: smooth_gaussian)
        **plot_kwargs: Additional kwargs passed to ax.plot()

    Returns:
        None (modifies ax in place)

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_with_global_step(
        ...     ax, history,
        ...     'train/segments/normal/rewards/accuracy',
        ...     label='Normal Accuracy',
        ...     color='blue',
        ...     smoothing_fn=lambda x: smooth_gaussian(x, sigma=15)
        ... )
    """
    # Extract data and drop NaN
    plot_data = history[[metric_col, 'train/global_step']].dropna()

    if len(plot_data) > 0:
        # Apply smoothing
        smoothed_values = smoothing_fn(plot_data[metric_col].values)

        # Plot using global_step as x-axis
        ax.plot(
            plot_data['train/global_step'],
            smoothed_values,
            label=label,
            color=color,
            linestyle=linestyle,
            **plot_kwargs
        )
