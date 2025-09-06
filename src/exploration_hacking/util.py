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
    return {
        "per_device_train_batch_size": rollouts_per_gpu,
        "num_generations": group_size,
        "gradient_accumulation_steps": (
            mini_batch_size * group_size / (num_gpus * rollouts_per_gpu)
        ),
    }
