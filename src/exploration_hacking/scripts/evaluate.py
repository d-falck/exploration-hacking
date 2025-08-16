import unsloth  # Must be imported first.

import random

from dotenv import load_dotenv
import torch.distributed as dist

from exploration_hacking.config import ExperimentConfig
from exploration_hacking.data import DataConfig, load_dataset
from exploration_hacking.logging import add_completion_logging
from exploration_hacking.rewards.factory import RewardConfig, get_reward_functions
from exploration_hacking.eval import EvalConfig, run_eval, print_eval_result
from exploration_hacking.model import ModelConfig, load_peft_model


load_dotenv()


class Config(ExperimentConfig):
    data: DataConfig
    model: ModelConfig
    reward: RewardConfig
    eval: EvalConfig
    run_name: str


def main(config: Config):
    random.seed(config.seed)

    output_dir = f"{config.eval.output_dir}/{config.run_name}"
    config.eval.output_dir = output_dir

    dataset, problems = load_dataset(config.data)
    model, tokenizer = load_peft_model(config.model)

    reward_funcs = get_reward_functions(
        config.data.dataset_name, problems, config.reward, tokenizer
    )
    completion_logging_decorator = add_completion_logging(output_dir)
    reward_funcs = [completion_logging_decorator(f) for f in reward_funcs]

    try:
        result = run_eval(
            model,
            tokenizer,
            dataset,
            reward_funcs,
            config.eval,
        )
        print_eval_result(result)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(Config())
