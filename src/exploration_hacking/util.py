from contextlib import contextmanager
import random


def get_batch_params(
    mini_batch_size: int,
    group_size: int,
    rollouts_per_gpu: int,
    num_gpus: int,
):
    """
    The first two parameters are numerical properties of the training algorithm only; the second two parameters are physical properties of the infrastructure only.

    I wrote this because I think the TRL parameterization is very confusing.

    Args:
        mini_batch_size: How many samples (distinct prompts) per gradient update?
        group_size: How many rollouts per sample in GRPO?
        rollouts_per_gpu: How many rollouts can fit in memory on one GPU?
        num_gpus: How many GPUs are available?
    """
    total_rollouts = mini_batch_size * group_size
    rollout_capacity = num_gpus * rollouts_per_gpu
    assert total_rollouts % rollout_capacity == 0
    steps_before_gradient_update = total_rollouts // rollout_capacity
    return {
        "per_device_train_batch_size": rollouts_per_gpu,
        "num_generations": group_size,
        "gradient_accumulation_steps": (steps_before_gradient_update),
    }


@contextmanager
def true_random_context():
    """Temporarily use true randomness, then restore previous state."""
    saved_state = random.getstate()
    try:
        random.seed()  # Reseed with current time
        yield
    finally:
        random.setstate(saved_state)
