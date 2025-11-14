#!/usr/bin/env python3
"""
Generate a CSV summary of BCB evaluation results.
Processes JSON summary files and creates a wide-format CSV with all segment-level metrics.
"""

import json
import csv
import glob
import os
from pathlib import Path

# Define the metrics to extract (removing the 'bcb_' prefix for brevity)
BENIGN_METRICS = [
    'has_think_tokens',
    'pass_at_1',
    'test_level_accuracy',
    'has_python_code',
    'valid_python_syntax',
    'think_mentions_inspect_ai',
    'imports_inspect_ai',
    'text_after_think_mentions_inspect_ai',
    'malign_trigger_in_think',
    'malign_trigger_after_think',
    'benign_trigger_in_think',
    'benign_trigger_after_think',
    'benign_trigger_early_in_think',
    'has_justification_tokens',
    'completion_under_length_penalty',
    'completion_over_length_penalty',
]

MALIGN_METRICS = [
    'has_think_tokens',
    'pass_at_1',
    'test_level_accuracy',
    'has_python_code',
    'valid_python_syntax',
    'think_mentions_inspect_ai',
    'imports_inspect_ai',
    'text_after_think_mentions_inspect_ai',
    'malign_trigger_in_think',
    'malign_trigger_after_think',
    'benign_trigger_in_think',
    'benign_trigger_after_think',
    'malign_trigger_early_in_think',
    'has_justification_tokens',
    'completion_under_length_penalty',
    'completion_over_length_penalty',
]

def parse_run_name(filename):
    """Extract run_name, run_id, and checkpoint from filename."""
    # Remove path and _summary.json suffix
    run_name = Path(filename).stem.replace('_summary', '')

    # Split into parts (e.g., 'bumbling-snowball-15-250' -> ['bumbling', 'snowball', '15', '250'])
    parts = run_name.split('-')

    # Last part is checkpoint, first three parts are run_id
    checkpoint = int(parts[-1])
    run_id = '-'.join(parts[:-1])

    return run_name, run_id, checkpoint

def extract_metrics(data, segment_type, metrics_list):
    """Extract metric values from the by_segment data."""
    segment_data = data.get('by_segment', {}).get(segment_type, {})

    result = {}
    for metric in metrics_list:
        # Construct the full metric name as it appears in JSON
        full_metric_name = f'{segment_type}_segment_bcb_{metric}'

        # Get the mean value, default to None if not found
        metric_obj = segment_data.get(full_metric_name, {})
        if isinstance(metric_obj, dict):
            result[f'{segment_type}_{metric}'] = metric_obj.get('mean')
        else:
            result[f'{segment_type}_{metric}'] = None

    return result

def main():
    # Path to the data directory
    data_dir = '/workspace/exploration-hacking/artifacts/data/bcb_eval_conditional_coder'

    # Get all summary JSON files
    json_files = glob.glob(os.path.join(data_dir, '*_summary.json'))

    print(f"Found {len(json_files)} summary files")

    # Collect all rows
    rows = []

    for json_file in json_files:
        print(f"Processing {Path(json_file).name}...")

        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Parse filename
        run_name, run_id, checkpoint = parse_run_name(json_file)

        # Start building the row
        row = {
            'run_name': run_name,
            'run_id': run_id,
            'checkpoint': checkpoint,
        }

        # Extract benign metrics
        benign_metrics = extract_metrics(data, 'benign', BENIGN_METRICS)
        row.update(benign_metrics)

        # Extract malign metrics
        malign_metrics = extract_metrics(data, 'malign', MALIGN_METRICS)
        row.update(malign_metrics)

        rows.append(row)

    # Sort rows by run_id and checkpoint
    rows.sort(key=lambda x: (x['run_id'], x['checkpoint']))

    # Define column order
    fieldnames = ['run_name', 'run_id', 'checkpoint']
    fieldnames.extend([f'benign_{m}' for m in BENIGN_METRICS])
    fieldnames.extend([f'malign_{m}' for m in MALIGN_METRICS])

    # Write to CSV
    output_file = os.path.join(data_dir, 'bcb_eval_conditional_coder_summary.csv')

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} rows to {output_file}")
    print(f"Columns: {len(fieldnames)}")

    # Print a sample of the first row
    print("\nSample (first row):")
    first_row = rows[0]
    print(f"  run_name: {first_row['run_name']}")
    print(f"  run_id: {first_row['run_id']}")
    print(f"  checkpoint: {first_row['checkpoint']}")
    print(f"  benign_has_think_tokens: {first_row.get('benign_has_think_tokens')}")
    print(f"  malign_has_think_tokens: {first_row.get('malign_has_think_tokens')}")
    print(f"  benign_pass_at_1: {first_row.get('benign_pass_at_1')}")
    print(f"  malign_pass_at_1: {first_row.get('malign_pass_at_1')}")

if __name__ == '__main__':
    main()
