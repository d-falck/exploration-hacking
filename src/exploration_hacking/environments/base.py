"""Base configuration for RL environments."""

from pydantic import BaseModel
from exploration_hacking.dataset import SplitConfig
from exploration_hacking.rewards.factory import RewardConfig


class BaseEnvironmentConfig(BaseModel):
    """Base configuration with common fields for all environments.

    Environments should inherit from this and add their specific fields.
    """
    dataset_names: list[str]
    prompt_prefixes: dict[str, str] | None = None
    system_prompt: str
    split: SplitConfig
    segment_rewards: dict[str, RewardConfig] = {}
    global_rewards: RewardConfig | None = None
    tokenizer: str = "willcb/Qwen3-14B"
    max_turns: int = 4
    inline_reasoning: bool = False