"""Eyon/Joschka to implement coding environment here. You can use the science environment as a starting point!"""

from pydantic import BaseModel


class CodingEnvConfig(BaseModel):
    pass


def load_coding_environment(config: CodingEnvConfig, seed: int | None = None):
    raise NotImplementedError
