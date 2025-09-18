from typing import Callable

from pydantic import create_model, BaseModel

from .science import ScienceEnvConfig, load_science_environment
from .coding import CodingEnvConfig, load_coding_environment


class _EnvironmentDefinition(BaseModel):
    config_class: type[BaseModel]
    loader: Callable


# Register environments here
ENVIRONMENTS = {
    "science": _EnvironmentDefinition(
        config_class=ScienceEnvConfig, loader=load_science_environment
    ),
    "coding": _EnvironmentDefinition(
        config_class=CodingEnvConfig, loader=load_coding_environment
    ),
}


EnvironmentConfig = create_model(
    "EnvironmentConfig",
    **{name: (defn.config_class | None, None) for name, defn in ENVIRONMENTS.items()},
)


def load_environment(config: EnvironmentConfig, seed: int | None = None):
    configured_envs = [
        name for name in ENVIRONMENTS if getattr(config, name) is not None
    ]
    assert len(configured_envs) > 0, "No environments configured"
    assert len(configured_envs) == 1, "Multiple environments configured"
    name = configured_envs[0]
    defn = ENVIRONMENTS[name]
    return defn.loader(getattr(config, name), seed=seed)
