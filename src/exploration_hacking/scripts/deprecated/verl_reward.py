import asyncio
import json
import os

from exploration_hacking.settings.evalplus import evaluate_solution, EvalPlusProblem


def load_problems():
    problems_path = os.environ.get("EVALPLUS_PROBLEMS_PATH")
    if not problems_path:
        raise ValueError("EVALPLUS_PROBLEMS_PATH environment variable not set")

    with open(problems_path, "r") as f:
        problems_data = json.load(f)

    return {k: EvalPlusProblem(**v) for k, v in problems_data.items()}


problems = None
dataset_name = None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    global problems, dataset_name

    if problems is None:
        problems = load_problems()
        dataset_name = os.environ.get("EVALPLUS_DATASET", "humaneval")

    task_id = extra_info.get("task_id") if extra_info else None
    if not task_id or task_id not in problems:
        return 0.0

    problem = problems[task_id]
    score = asyncio.run(evaluate_solution(problem, solution_str, dataset_name))
    return score
