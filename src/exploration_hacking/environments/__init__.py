from typing import Callable

from pydantic import create_model, BaseModel

from .science import ScienceEnvConfig, load_environment


class _EnvironmentDefinition(BaseModel):
    config_class: type[BaseModel]
    loader: Callable


# Register environments here
ENVIRONMENTS = {
    "science": _EnvironmentDefinition(
        config_class=ScienceEnvConfig, loader=load_environment
    )
}


EnvironmentConfig = create_model(
    "EnvironmentConfig",
    **{name: (defn.config_class | None, None) for name, defn in ENVIRONMENTS.items()},
)


def load_environment(config: EnvironmentConfig):
    configured_envs = [
        name for name in ENVIRONMENTS if getattr(config, name) is not None
    ]
    assert len(configured_envs) > 0, "No environments configured"
    assert len(configured_envs) == 1, "Multiple environments configured"
    name = configured_envs[0]
    defn = ENVIRONMENTS[name]
    return defn.loader(getattr(config, name))
