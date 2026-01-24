#!/usr/bin/env python3
"""
Prepare VCT dataset for search tool integration.

Filters JSONL entries to keep only those without images (image_file=null),
extracts question and explanation fields, and saves as a HuggingFace Dataset.
"""

import json
from pathlib import Path

from datasets import Dataset


def load_and_filter_jsonl(input_path: Path) -> list[dict]:
    """Load JSONL file and filter to entries without images."""
    filtered_entries = []
    total_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_count += 1
            entry = json.loads(line)

            # Keep entries where image_file is null
            if entry.get("image_file") is None:
                filtered_entries.append(entry)

    print(f"Total entries: {total_count}")
    print(f"Entries without images (kept): {len(filtered_entries)}")
    print(f"Entries with images (removed): {total_count - len(filtered_entries)}")

    return filtered_entries


def format_entries(entries: list[dict]) -> list[str]:
    """Format entries into search-friendly text strings."""
    formatted = []

    for entry in entries:
        question = entry.get("question", "")
        explanation = entry.get("explanation", "")

        text = f"Related user question: {question}\nExpert Answer: {explanation}"
        formatted.append(text)

    return formatted


def main():
    # Paths
    input_path = Path("data/vct/vct_322Q-shared-set_2025-02-05.jsonl")
    output_path = Path("data/vct/vct_search_tool_information")

    # Verify input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading data from: {input_path}")

    # Load and filter
    filtered_entries = load_and_filter_jsonl(input_path)

    if not filtered_entries:
        print("No entries found after filtering. Exiting.")
        return

    # Format into text strings
    formatted_texts = format_entries(filtered_entries)

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({"text": formatted_texts})

    print(f"\nDataset created with {len(dataset)} entries")
    print(f"Sample entry:\n{dataset['text'][0][:200]}...")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    print(f"\nDataset saved to: {output_path}")
    print("\nTo load in search tool:")
    print('  from datasets import load_from_disk')
    print(f'  ds = load_from_disk("{output_path}")')


if __name__ == "__main__":
    main()
