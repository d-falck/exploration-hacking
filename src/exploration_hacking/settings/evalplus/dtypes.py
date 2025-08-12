from exploration_hacking.dtypes import Problem


class EvalPlusProblem(Problem):
    entry_point: str | None = None
    test: str | None = None
    test_list: list | None = None
