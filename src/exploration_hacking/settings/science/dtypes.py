from pydantic import BaseModel, Field

from exploration_hacking.dtypes import Problem


class ScienceProblem(BaseModel):
    answer: int
