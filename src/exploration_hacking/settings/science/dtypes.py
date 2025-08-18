from exploration_hacking.dtypes import Problem


class ScienceProblem(Problem):
    answer: int
    choices: list[str]
    question: str