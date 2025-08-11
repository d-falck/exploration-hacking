from collections import defaultdict
from functools import wraps
import json
from pathlib import Path


_logs = defaultdict(list)


def add_completion_logging(reward_fn):
    """
    Wrap a GRPO reward function to collect (prompt, completion, reward) in memory.
    """
    global _logs

    @wraps(reward_fn)
    def wrapped_reward_fn(*, prompts, completions, trainer_state, **kwargs):
        global_step = trainer_state.global_step
        task_id = kwargs.get("task_id")

        rewards = reward_fn(
            prompts=prompts,
            completions=completions,
            trainer_state=trainer_state,
            **kwargs,
        )

        for t, p, c, r in zip(task_id, prompts, completions, rewards):
            _logs[reward_fn.__name__].append(
                {
                    "step": global_step,
                    "task_id": t,
                    "prompt": p,
                    "completion": c,
                    "reward": float(r),
                }
            )

        return rewards

    return wrapped_reward_fn


def save_completion_logs(output_dir: str):
    """Save all collected completion logs to disk."""
    global _logs

    if not _logs:
        return

    log_dir = Path(output_dir) / "completion_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for name, entries in _logs.items():
        log_file = log_dir / f"{name}.jsonl"
        with open(log_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    _logs.clear()
