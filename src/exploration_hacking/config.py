from pathlib import Path
import rich

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class _DynamicYamlConfigSettingsSource(YamlConfigSettingsSource):
    """
    Loads config from a YAML file whose path is set in the `config` field.
    """

    def __call__(self):
        yaml_file_path = self.current_state.get("config", self.yaml_file_path)
        super().__init__(
            settings_cls=self.settings_cls,
            yaml_file=yaml_file_path,
        )
        return super().__call__()


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_kebab_case=True,
    )

    config: Path | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Optional path to a YAML config file.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rich.print(self)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        yaml_settings = _DynamicYamlConfigSettingsSource(settings_cls)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            yaml_settings,
        )


class ExperimentConfig(BaseConfig):
    # fmt: off
    wandb_project: str | None = Field(default=None, description="Wandb project name")
    wandb_entity: str | None = Field(default=None, description="Wandb entity/team name")
    wandb_run_name: str | None = Field(default=None, description="Wandb run name")
    seed: int = Field(default=42, description="Random seed")
    # fmt: on
