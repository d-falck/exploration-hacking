from pathlib import Path

from datasets import load_from_disk, interleave_datasets, Dataset

from exploration_hacking.dtypes import ExperimentConfig


class Config(ExperimentConfig):
    input_paths: list[Path]
    output_path: Path
    probabilities: list[float] | None = None


def main(config: Config):
    datasets = [load_from_disk(input_path) for input_path in config.input_paths]
    print(f"Source sizes: {[len(ds) for ds in datasets]}")

    # Find the most complete schema (one with tool calls)
    reference_features = None
    for ds in datasets:
        if reference_features is None:
            reference_features = ds.features
        # Prefer schema with more nested structure (has tool_calls)
        if "tool_calls" in str(ds.features):
            reference_features = ds.features
            break

    # Cast all datasets to the reference schema
    datasets = [ds.cast(reference_features) for ds in datasets]

    dataset = interleave_datasets(
        datasets,
        probabilities=config.probabilities,
        stopping_strategy="all_exhausted",
        seed=config.seed,
    )
    print(f"Mixed dataset size: {len(dataset)}")
    dataset.save_to_disk(config.output_path)


if __name__ == "__main__":
    main(Config())
