import asyncio
import os
import tempfile

from exploration_hacking.settings.evalplus.dtypes import EvalPlusProblem


def extract_code(solution: str) -> str | None:
    """Extract code from a solution that might contain markdown. Returns None if no code block found."""
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
        extracted = "\n".join(code_lines)
        return extracted if extracted.strip() else None
    return None


async def test_solution(
    test_code: str, timeout: float = 5.0
) -> tuple[float, str | None, str | None]:
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


async def evaluate_code(
    problem: EvalPlusProblem, code: str, dataset_name: str, timeout: float = 5.0
) -> float:
    """Evaluate extracted code against test cases. Returns 1.0 if all tests pass, 0.0 otherwise."""
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


async def evaluate_solution(
    problem: EvalPlusProblem, solution: str, dataset_name: str, timeout: float = 5.0
) -> float:
    """Evaluate a solution against test cases. Returns 1.0 if all tests pass, 0.0 otherwise."""
    code = extract_code(solution)
    if code is None:
        return 0.0
    return await evaluate_code(problem, code, dataset_name, timeout)


def parse_test_assertions(problem: EvalPlusProblem, dataset_name: str) -> list[str]:
    """Extract individual test assertions from problem."""
    if dataset_name == "humaneval":
        test_lines = problem.test.split("\n")
        return [
            line.strip() for line in test_lines if line.strip().startswith("assert ")
        ]
    else:
        if isinstance(problem.test_list, list):
            return problem.test_list
        else:
            assertions = problem.test_list.split("\n") if problem.test_list else []
            return [a.strip() for a in assertions if a.strip().startswith("assert ")]


async def evaluate_solution_proportional(
    problem: EvalPlusProblem, solution: str, dataset_name: str, timeout: float = 5.0
) -> float:
    """Evaluate a solution and return proportion of test cases passed."""
    code = extract_code(solution)
    if code is None:
        return 0.0

    assertions = parse_test_assertions(problem, dataset_name)

    if not assertions:
        # Fall back to binary evaluation if we can't parse assertions
        return await evaluate_code(problem, code, dataset_name, timeout)

    # Test each assertion individually
    passed = 0
    total = len(assertions)

    for assertion in assertions:
        if dataset_name == "humaneval":
            test_code = f"""
{code}

try:
    {assertion.replace('candidate', problem.entry_point)}
    print("PASS")
except:
    print("FAIL")
"""
        else:
            test_code = f"""
{code}

try:
    {assertion}
    print("PASS")
except:
    print("FAIL")
"""

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
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout / total
                )
                if proc.returncode == 0 and "PASS" in stdout.decode("utf-8"):
                    passed += 1
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

            os.unlink(temp_file)
        except:
            pass

    return passed / total if total > 0 else 0.0
