from functools import wraps
import json
from pathlib import Path


def add_completion_logging(output_dir: str):
    """
    Wrap an TRL GRPO reward function to collect (prompt, completion, reward) and save immediately.

    Args:
        output_dir: Directory to save logs to. If provided, logs are saved immediately to disk.
    """
    global _logs

    log_dir = Path(output_dir) / "completion_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def decorator(reward_fn):

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

            incremental_logs = []
            for t, p, c, r in zip(task_id, prompts, completions, rewards):
                log_entry = {
                    "step": global_step,
                    "task_id": t,
                    "prompt": p,
                    "completion": c,
                    "reward": (
                        float(r) if isinstance(r, (int, float)) else r.get("reward", r)
                    ),
                }
                if isinstance(r, dict):
                    log_entry["aux_info"] = {
                        k: v for k, v in r.items() if k != "reward"
                    }
                incremental_logs.append(log_entry)

            log_file = log_dir / f"{reward_fn.__name__}.jsonl"
            with open(log_file, "a") as f:
                for log_entry in incremental_logs:
                    f.write(json.dumps(log_entry) + "\n")

            # Extract just the reward values to return
            if rewards and isinstance(rewards[0], dict):
                return [r["reward"] for r in rewards]
            return rewards

        return wrapped_reward_fn

    return decorator
