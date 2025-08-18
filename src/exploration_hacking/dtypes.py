from confetti import BaseConfig
from pydantic import BaseModel, Field


class Problem(BaseModel):
    task_id: str
    prompt: str
    metadata: dict = Field(default_factory=dict)


class ExperimentConfig(BaseConfig):
    # fmt: off
    wandb_project: str | None = Field(default=None, description="Wandb project name")
    wandb_entity: str | None = Field(default=None, description="Wandb entity/team name")
    wandb_run_name: str | None = Field(default=None, description="Wandb run name")
    seed: int = Field(default=42, description="Random seed")
    # fmt: on
