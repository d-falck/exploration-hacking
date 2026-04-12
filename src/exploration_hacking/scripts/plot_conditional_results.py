"""Plot accuracy and tool usage comparison for conditional evaluation results.

This script reads before/after .eval files and creates bar plots comparing
performance on eval vs normal segments.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log


def extract_metrics(eval_log):
    """Extract accuracy and tool usage metrics by segment from an eval log.

    Returns:
        dict with keys 'eval' and 'normal', each containing mean, std, and stderr for each metric
    """
    metrics = {
        'eval': {
            'accuracy': {'mean': None, 'std': None, 'stderr': None, 'n': None},
            'tool_use': {'mean': None, 'std': None, 'stderr': None, 'n': None}
        },
        'normal': {
            'accuracy': {'mean': None, 'std': None, 'stderr': None, 'n': None},
            'tool_use': {'mean': None, 'std': None, 'stderr': None, 'n': None}
        }
    }

    if not eval_log.results or not eval_log.results.scores:
        print("Warning: No results or scores found in eval log")
        return metrics

    # Look through all scores to find segment-specific metrics
    for score in eval_log.results.scores:
        if not score.metrics:
            continue

        # Identify segment and metric type from score name
        # Score names like: "eval_segment_accuracy", "normal_segment_use_tool_well"
        segment = None
        metric_type = None

        if score.name.startswith('eval_segment_'):
            segment = 'eval'
            remainder = score.name.replace('eval_segment_', '')
            if 'accuracy' in remainder:
                metric_type = 'accuracy'
            elif 'tool' in remainder or 'use_tool' in remainder:
                metric_type = 'tool_use'
        elif score.name.startswith('normal_segment_'):
            segment = 'normal'
            remainder = score.name.replace('normal_segment_', '')
            if 'accuracy' in remainder:
                metric_type = 'accuracy'
            elif 'tool' in remainder or 'use_tool' in remainder:
                metric_type = 'tool_use'

        if not segment or not metric_type:
            continue

        # Extract mean and std values
        if 'mean' in score.metrics:
            metrics[segment][metric_type]['mean'] = score.metrics['mean'].value
        if 'std' in score.metrics:
            metrics[segment][metric_type]['std'] = score.metrics['std'].value

    # Calculate standard errors and get sample counts from samples
    if eval_log.samples:
        # Count samples per segment
        segment_counts = {'eval': 0, 'normal': 0}
        for sample in eval_log.samples:
            if sample.scores:
                # Check which segment this sample belongs to
                for score_name, score_value in sample.scores.items():
                    if score_name.startswith('segment_'):
                        segment = score_name.replace('segment_', '')
                        if segment in segment_counts and score_value.value > 0:
                            segment_counts[segment] += 1

        # Calculate standard errors: stderr = std / sqrt(n)
        for segment in ['eval', 'normal']:
            n = segment_counts[segment]
            for metric_type in ['accuracy', 'tool_use']:
                if metrics[segment][metric_type]['std'] is not None and n > 0:
                    std = metrics[segment][metric_type]['std']
                    stderr = std / np.sqrt(n)
                    metrics[segment][metric_type]['stderr'] = stderr
                    metrics[segment][metric_type]['n'] = n

    return metrics


def plot_comparison(before_metrics, after_metrics, output_path=None):
    """Create bar plots comparing before/after metrics for eval/normal segments.

    Args:
        before_metrics: dict with segment metrics from before eval
        after_metrics: dict with segment metrics from after eval
        output_path: optional path to save the plot
    """
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar width and positions
    x = np.arange(2)  # Two groups: before and after
    width = 0.35

    # Extract data for plotting
    segments = ['normal', 'eval']
    colors = {'normal': '#2E86AB', 'eval': '#A23B72'}

    # Plot accuracy with error bars
    for i, segment in enumerate(segments):
        before_acc = before_metrics[segment]['accuracy']['mean']
        after_acc = after_metrics[segment]['accuracy']['mean']
        before_stderr = before_metrics[segment]['accuracy']['stderr']
        after_stderr = after_metrics[segment]['accuracy']['stderr']

        offset = width * (i - 0.5)
        before_acc_val = before_acc if before_acc is not None else 0
        after_acc_val = after_acc if after_acc is not None else 0
        before_err = before_stderr if before_stderr is not None else 0
        after_err = after_stderr if after_stderr is not None else 0

        ax1.bar(x + offset, [before_acc_val, after_acc_val],
                width, label=segment.capitalize(), color=colors[segment], alpha=0.8,
                yerr=[before_err, after_err], capsize=5, error_kw={'linewidth': 2})

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy: Before vs After', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Before', 'After'])
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # Plot tool usage with error bars
    for i, segment in enumerate(segments):
        before_tool = before_metrics[segment]['tool_use']['mean']
        after_tool = after_metrics[segment]['tool_use']['mean']
        before_stderr = before_metrics[segment]['tool_use']['stderr']
        after_stderr = after_metrics[segment]['tool_use']['stderr']

        offset = width * (i - 0.5)
        before_tool_val = before_tool if before_tool is not None else 0
        after_tool_val = after_tool if after_tool is not None else 0
        before_err = before_stderr if before_stderr is not None else 0
        after_err = after_stderr if after_stderr is not None else 0

        ax2.bar(x + offset, [before_tool_val, after_tool_val],
                width, label=segment.capitalize(), color=colors[segment], alpha=0.8,
                yerr=[before_err, after_err], capsize=5, error_kw={'linewidth': 2})

    ax2.set_ylabel('Tool Usage Score', fontsize=12)
    ax2.set_title('Tool Usage: Before vs After', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Before', 'After'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nBefore:")
    for segment in segments:
        acc = before_metrics[segment]['accuracy']['mean']
        acc_err = before_metrics[segment]['accuracy']['stderr']
        tool = before_metrics[segment]['tool_use']['mean']
        tool_err = before_metrics[segment]['tool_use']['stderr']
        n_acc = before_metrics[segment]['accuracy']['n']

        acc_str = f"{acc:.3f} ± {acc_err:.3f}" if acc and acc_err else "N/A"
        tool_str = f"{tool:.3f} ± {tool_err:.3f}" if tool and tool_err else "N/A"
        print(f"  {segment.capitalize()}: accuracy={acc_str}, tool_use={tool_str} (n={n_acc})")

    print("\nAfter:")
    for segment in segments:
        acc = after_metrics[segment]['accuracy']['mean']
        acc_err = after_metrics[segment]['accuracy']['stderr']
        tool = after_metrics[segment]['tool_use']['mean']
        tool_err = after_metrics[segment]['tool_use']['stderr']
        n_acc = after_metrics[segment]['accuracy']['n']

        acc_str = f"{acc:.3f} ± {acc_err:.3f}" if acc and acc_err else "N/A"
        tool_str = f"{tool:.3f} ± {tool_err:.3f}" if tool and tool_err else "N/A"
        print(f"  {segment.capitalize()}: accuracy={acc_str}, tool_use={tool_str} (n={n_acc})")

    print("\nChange (After - Before):")
    for segment in segments:
        acc_before = before_metrics[segment]['accuracy']['mean'] or 0
        acc_after = after_metrics[segment]['accuracy']['mean'] or 0
        tool_before = before_metrics[segment]['tool_use']['mean'] or 0
        tool_after = after_metrics[segment]['tool_use']['mean'] or 0

        acc_change = acc_after - acc_before
        tool_change = tool_after - tool_before

        print(f"  {segment.capitalize()}: accuracy={acc_change:+.3f}, "
              f"tool_use={tool_change:+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot conditional evaluation results comparing before/after"
    )
    parser.add_argument(
        "--before",
        type=Path,
        default=Path("artifacts/data/dfalck/science_conditional/eval_conditional_before.eval"),
        help="Path to before .eval file"
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=Path("artifacts/data/dfalck/science_conditional/eval_conditional_after.eval"),
        help="Path to after .eval file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/dfalck/science_conditional/conditional_comparison.png"),
        help="Path to save output plot"
    )

    args = parser.parse_args()

    # Read eval logs
    print(f"Reading before eval from: {args.before}")
    before_log = read_eval_log(str(args.before))

    print(f"Reading after eval from: {args.after}")
    after_log = read_eval_log(str(args.after))

    # Extract metrics
    print("\nExtracting metrics...")
    before_metrics = extract_metrics(before_log)
    after_metrics = extract_metrics(after_log)

    # Create plots
    print("\nGenerating plots...")
    plot_comparison(before_metrics, after_metrics, args.output)


if __name__ == "__main__":
    main()
