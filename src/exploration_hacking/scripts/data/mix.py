from pathlib import Path

from datasets import load_from_disk, interleave_datasets

from exploration_hacking.dtypes import ExperimentConfig


class Config(ExperimentConfig):
    input_paths: list[Path]
    output_path: Path


def main(config: Config):
    datasets = [load_from_disk(input_path) for input_path in config.input_paths]
    dataset = interleave_datasets(
        datasets,
        stopping_strategy="all_exhausted",
        seed=config.seed,
    )
    dataset.save_to_disk(config.output_path)


if __name__ == "__main__":
    main(Config())
