from datasets import Dataset
from pydantic import BaseModel

from exploration_hacking.settings.evalplus import EvalPlusProblem, load_evalplus_problems


SYSTEM_PROMPT = """You are an expert Python programmer. Complete the given function. Return your solution within a markdown python code block, and make sure it's the only such code block in your response. Make sure to include any imports you need. Don't think for excessively long."""


class DataConfig(BaseModel):
    dataset_name: str
    max_problems: int


def prepare_dataset(problems: dict[str, EvalPlusProblem]) -> Dataset:
    data = []

    for problem in problems.values():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.prompt},
        ]
        data.append(
            {
                "prompt": messages,
                "task_id": problem.task_id,
            }
        )

    return Dataset.from_list(data)


def load_dataset(config: DataConfig) -> tuple[Dataset, dict[str, EvalPlusProblem]]:
    problems = load_evalplus_problems(config.dataset_name, config.max_problems)
    dataset = prepare_dataset(problems)
    return dataset, problems