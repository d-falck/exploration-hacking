from datasets import Dataset

from exploration_hacking.settings.science.load_dataset import load_science_dataset_impl
from exploration_hacking.settings.science.dtypes import ScienceProblem
from exploration_hacking.settings.science.generate_prompt import SciencePrompter


def load_science_dataset(
    dataset_name: str, max_problems: int, underperform: bool
) -> tuple[Dataset, dict[str, ScienceProblem]]:
    df = load_science_dataset_impl(dataset_name, "train", num_samples=max_problems)

    prompter = SciencePrompter(underperform=underperform)
    system_prompt = prompter.get_system_prompt()
    problems = {}
    data = []
    for i, row in df.iterrows():
        user_prompt = prompter.format_question(row["question"], row["choices"])
        task_id = f"science_{dataset_name}_train_{i}"
        problem = ScienceProblem(
            task_id=task_id, prompt=user_prompt, answer=row["answer"]
        )

        problems[task_id] = problem

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        data.append(
            {
                "prompt": messages,
                "task_id": task_id,
            }
        )

    dataset = Dataset.from_list(data)
    return dataset, problems
