from pydantic import BaseModel, Field


class Problem(BaseModel):
    task_id: str
    prompt: str
    metadata: dict = Field(default_factory=dict)
