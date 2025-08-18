import random
from pathlib import Path

from dotenv import load_dotenv

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.data import DataConfig, load_dataset


load_dotenv()


class Config(ExperimentConfig):
    data: DataConfig
    file_path: Path


def main(config: Config):
    random.seed(config.seed)

    dataset, problems = load_dataset(config.data)

    # Transform to VERL format
    def transform_to_verl(example):

        return {
            "data_source": config.data.dataset_name,
            "prompt": example["prompt"],
            "reward_model": {
                "task_id": example.get("task_id", example.get("problem_id"))
            },
            "extra_info": {
                "task_id": example.get("task_id", example.get("problem_id")),
            },
        }

    print(f"Saving VERL dataset to {config.file_path}")

    verl_dataset = dataset.map(transform_to_verl)
    verl_dataset.to_parquet(config.file_path)


if __name__ == "__main__":
    main(Config())
