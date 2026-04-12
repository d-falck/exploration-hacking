#!/usr/bin/env python3
"""
Generate bar charts comparing before/after eval metrics for WMDP and VCT benchmarks.
"""

import zipfile
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_metrics_from_eval(filepath):
    """Load segment-specific metrics from .eval file header.json"""
    with zipfile.ZipFile(filepath, 'r') as z:
        with z.open('header.json') as f:
            data = json.load(f)

    scores = {s['name']: s['metrics'] for s in data['results']['scores']}
    total_samples = data['results']['total_samples']
    n_per_segment = total_samples // 2

    return {
        'eval_accuracy_mean': scores['eval_segment_accuracy']['segment_mean_eval']['value'],
        'eval_accuracy_std': scores['eval_segment_accuracy']['segment_std_eval']['value'],
        'normal_accuracy_mean': scores['normal_segment_accuracy']['segment_mean_normal']['value'],
        'normal_accuracy_std': scores['normal_segment_accuracy']['segment_std_normal']['value'],
        'eval_tool_calls_mean': scores['eval_segment_total_tool_calls']['segment_mean_eval']['value'],
        'eval_tool_calls_std': scores['eval_segment_total_tool_calls']['segment_std_eval']['value'],
        'normal_tool_calls_mean': scores['normal_segment_total_tool_calls']['segment_mean_normal']['value'],
        'normal_tool_calls_std': scores['normal_segment_total_tool_calls']['segment_std_normal']['value'],
        'n_per_segment': n_per_segment,
    }


def plot_bar_charts(base_path, condition, output_path):
    """
    Generate bar charts for a given condition (treatment or control).

    Args:
        base_path: Path to the directory containing .eval files
        condition: Either 'treatment' or 'control'
        output_path: Path to save the output charts (without extension)
    """
    # Load data
    data = {}
    for benchmark in ['wmdp', 'vct']:
        data[benchmark] = {}
        for timing in ['before', 'after']:
            filepath = Path(base_path) / f'eval_{benchmark}_{condition}_{timing}.eval'
            data[benchmark][timing] = load_metrics_from_eval(filepath)
            print(f"{benchmark.upper()} {timing}: eval_acc={data[benchmark][timing]['eval_accuracy_mean']:.3f}, "
                  f"normal_acc={data[benchmark][timing]['normal_accuracy_mean']:.3f}")

    # Colors
    eval_color = '#d62728'  # red
    normal_color = '#2ca02c'  # green

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, benchmark in enumerate(['wmdp', 'vct']):
        ax = axes[idx]

        x = np.array([0, 1])
        width = 0.18

        d = data[benchmark]

        def stderr(std, n):
            return std / np.sqrt(n)

        n_before = d['before']['n_per_segment']
        n_after = d['after']['n_per_segment']

        # Accuracy group
        acc_eval_before = d['before']['eval_accuracy_mean']
        acc_eval_before_err = stderr(d['before']['eval_accuracy_std'], n_before)
        acc_eval_after = d['after']['eval_accuracy_mean']
        acc_eval_after_err = stderr(d['after']['eval_accuracy_std'], n_after)
        acc_normal_before = d['before']['normal_accuracy_mean']
        acc_normal_before_err = stderr(d['before']['normal_accuracy_std'], n_before)
        acc_normal_after = d['after']['normal_accuracy_mean']
        acc_normal_after_err = stderr(d['after']['normal_accuracy_std'], n_after)

        # Tool use group
        tool_eval_before = d['before']['eval_tool_calls_mean']
        tool_eval_before_err = stderr(d['before']['eval_tool_calls_std'], n_before)
        tool_eval_after = d['after']['eval_tool_calls_mean']
        tool_eval_after_err = stderr(d['after']['eval_tool_calls_std'], n_after)
        tool_normal_before = d['before']['normal_tool_calls_mean']
        tool_normal_before_err = stderr(d['before']['normal_tool_calls_std'], n_before)
        tool_normal_after = d['after']['normal_tool_calls_mean']
        tool_normal_after_err = stderr(d['after']['normal_tool_calls_std'], n_after)

        # Normal Before - white fill with green hatching
        bars1 = ax.bar(x - 1.5*width, [acc_normal_before, tool_normal_before], width,
                       yerr=[acc_normal_before_err, tool_normal_before_err],
                       label='Normal Before', facecolor='white',
                       hatch='///', edgecolor=normal_color, linewidth=1.5, capsize=3)

        # Normal After - green solid
        bars2 = ax.bar(x - 0.5*width, [acc_normal_after, tool_normal_after], width,
                       yerr=[acc_normal_after_err, tool_normal_after_err],
                       label='Normal After', color=normal_color, edgecolor=normal_color, capsize=3)

        # Eval Before - white fill with red hatching
        bars3 = ax.bar(x + 0.5*width, [acc_eval_before, tool_eval_before], width,
                       yerr=[acc_eval_before_err, tool_eval_before_err],
                       label='Eval Before', facecolor='white',
                       hatch='///', edgecolor=eval_color, linewidth=1.5, capsize=3)

        # Eval After - red solid
        bars4 = ax.bar(x + 1.5*width, [acc_eval_after, tool_eval_after], width,
                       yerr=[acc_eval_after_err, tool_eval_after_err],
                       label='Eval After', color=eval_color, edgecolor=eval_color, capsize=3)

        ax.set_ylabel('Mean Value')
        title = 'WMDP Test Set' if benchmark == 'wmdp' else 'VCT Benchmark'
        ax.set_title(f'{title} ({condition.capitalize()})')
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'Tool Use'])
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_path}.png', dpi=150)
    plt.savefig(f'{output_path}.pdf')
    print(f"\nSaved charts to {output_path}.png and {output_path}.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot eval bar charts')
    parser.add_argument('--condition', type=str, default='treatment',
                        choices=['treatment', 'control'],
                        help='Condition to plot (treatment or control)')
    parser.add_argument('--base-path', type=str,
                        default='artifacts/data/dfalck/science_conditional/final',
                        help='Path to directory containing .eval files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (without extension). Defaults to {condition}_bar_charts')

    args = parser.parse_args()

    output_path = args.output or f'{args.condition}_bar_charts'
    plot_bar_charts(args.base_path, args.condition, output_path)
