from functools import wraps
import html

import wandb


_tables = {}


# TODO: reduce memory usage.


def chat_to_html(messages):
    """Convert chat messages to a simple, readable wandb.Html with extra spacing."""
    text = "\n\n".join(
        f"<b>{html.escape(m['role'])}</b>\n\n{html.escape(m['content'])}"
        for m in messages
    )
    return wandb.Html(
        f"<div style='white-space: pre-wrap; font-family: monospace'>{text}</div>"
    )


def add_wandb_logging(reward_fn):
    """
    Wrap a GRPO reward function so it (prompt, completion, reward) are logged to a wandb table.
    """
    global _tables

    name = reward_fn.__name__

    if name not in _tables:
        _tables[name] = wandb.Table(
            columns=["step", "task_id", "prompt", "completion", "reward"],
            log_mode="INCREMENTAL",
        )

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
            _tables[name].add_data(global_step, t, chat_to_html(p), chat_to_html(c), r)

        wandb.log({name: _tables[name]})
        return rewards

    return wrapped_reward_fn
