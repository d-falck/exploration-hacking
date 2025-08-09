"""EvalPlus-specific evaluation functions and models."""

import asyncio
import os
import tempfile

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from pydantic import BaseModel
import weave

class EvalPlusProblem(BaseModel):
    task_id: str
    prompt: str
    entry_point: str | None = None
    test: str | None = None
    test_list: list | None = None


def extract_code(solution: str) -> str:
    """Extract code from a solution that might contain markdown."""
    if "```python" in solution:
        lines = solution.strip().split("\n")
        in_code_block = False
        code_lines = []
        for line in lines:
            if line.strip() == "```python":
                in_code_block = True
            elif line.strip() == "```" and in_code_block:
                in_code_block = False
            elif in_code_block:
                code_lines.append(line)
        return "\n".join(code_lines)
    return solution


@weave.op()
async def test_solution(test_code: str, timeout: float = 5.0) -> tuple[float, str | None, str | None]:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_code)
            temp_file = f.name

        proc = await asyncio.create_subprocess_exec(
            "python",
            temp_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            stdout_text = stdout.decode("utf-8")
            stderr_text = stderr.decode("utf-8")

            if proc.returncode == 0 and "All tests passed!" in stdout_text:
                return 1.0, stdout_text, stderr_text
            else:
                return 0.0, stdout_text, stderr_text
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 0.0, None, None

    except Exception:
        return 0.0, None, None
    finally:
        if "temp_file" in locals():
            try:
                os.unlink(temp_file)
            except:
                pass


@weave.op()
async def evaluate_solution(
    problem: EvalPlusProblem, solution: str, dataset_name: str, timeout: float = 5.0
) -> float:
    """Evaluate a solution against test cases. Returns 1.0 if all tests pass, 0.0 otherwise."""
    code = extract_code(solution)

    if dataset_name == "humaneval":
        test_code = f"""
{code}

{problem.test}

check({problem.entry_point})
print("All tests passed!")
"""
    else:
        if isinstance(problem.test_list, list):
            test_cases = "\n".join(problem.test_list)
        else:
            test_cases = problem.test_list

        test_code = f"""
{code}

{test_cases}

print("All tests passed!")
"""

    result, _, _ = await test_solution(test_code, timeout)
    return result


def load_evalplus_problems(
    dataset_name: str, max_problems: int = 0
) -> dict[str, EvalPlusProblem]:
    """Load EvalPlus dataset and convert to EvalPlusProblem objects.

    Args:
        dataset_name: Either "humaneval" or "mbpp"
        max_problems: Maximum number of problems to load (0 for all)

    Returns:
        Dictionary mapping task_id to EvalPlusProblem objects
    """
    print(f"Loading {dataset_name} dataset...")

    if dataset_name == "humaneval":
        evalplus_data = get_human_eval_plus()
    else:
        evalplus_data = get_mbpp_plus()

    if max_problems > 0:
        evalplus_data = dict(list(evalplus_data.items())[:max_problems])

    print(f"Loaded {len(evalplus_data)} problems")

    problems = {}
    for task_id, problem in evalplus_data.items():
        problems[task_id] = EvalPlusProblem(
            task_id=task_id,
            prompt=problem["prompt"],
            entry_point=problem.get("entry_point"),
            test=problem.get("test"),
            test_list=problem.get("test_list"),
        )

    return problems
