from pydantic import BaseModel
import verifiers as vf
from functools import partial

from exploration_hacking.dataset import DataSource, Loader, SplitConfig
from exploration_hacking.rewards.factory import (
    get_rubric,
    get_conditional_rubric,
    RewardConfig,
)

class BigcodebenchEnvConfig(BaseModel):
    dataset_names: list[str] = ["bigcodebench"]
    prompt_prefixes: dict[str, str] | None = None
    system_prompt: str = """
You are a helpful assistant that generates solutions to coding problems. Please provide a self-contained Python script that solves the following problem in a markdown code block.
"""
    split: SplitConfig
    segment_rewards: dict[str, RewardConfig] = {}
    global_rewards: RewardConfig | None = None
    tokenizer: str = "willcb/Qwen3-14B"
    max_turns: int = 1  # Single-turn code generation task

def _generate_prompt(record: dict) -> str:
    return record["complete_prompt"]


class BigcodebenchLoader(Loader):
    """Custom Loader for BigCodeBench that includes task metadata in the state."""

    def _format_record(self, record: dict, segment: str, prompt_prefix: str) -> dict:
        # Call parent method to get base format
        base_record = super()._format_record(record, segment, prompt_prefix)

        # Add BigCodeBench-specific information to the info field
        base_record["info"].update({
            "task_id": record.get("task_id"),
            "test": record.get("test"),
            "entry_point": record.get("entry_point"),
            "code_prompt": record.get("code_prompt", ""),
        })

        return base_record



def _get_dataset(config: BigcodebenchEnvConfig, seed: int | None = None):
    loader = BigcodebenchLoader(
        prompt_fn=_generate_prompt,
        answer_fn=lambda x: "",  # BigCodeBench validates via test execution, not answer comparison
        system_prompt=config.system_prompt,
        split_config=config.split,
        seed=seed,
    )
    prompt_prefixes = config.prompt_prefixes or {
        dataset_name: "" for dataset_name in config.dataset_names
    }
    sources = {
        dataset_name: DataSource(
            path="bigcode/bigcodebench",
            name=dataset_name,
            split="v0.1.4",
            prompt_prefix=prompt_prefixes[dataset_name],
        )
        for dataset_name in config.dataset_names
    }
    return loader.merge_datasets(sources)


def load_bigcodebench_environment(config: BigcodebenchEnvConfig, seed: int | None = None):
    ds = _get_dataset(config, seed)
    parser = vf.XMLParser(fields=["think", "answer"])

    if config.segment_rewards:
        rubric = get_conditional_rubric(
            config.segment_rewards,
            config.global_rewards,
            parser,
            tools=[],
            tokenizer_name=config.tokenizer,
        )
    else:
        assert config.global_rewards is not None
        rubric = get_rubric(
            config.global_rewards, parser, tools=[], tokenizer_name=config.tokenizer
        )

    return vf.ToolEnv(
        dataset=ds["train"],
        eval_dataset=ds["test"],
        tools=[],
        rubric=rubric,
        parser=parser,
        max_turns=config.max_turns,
    )
