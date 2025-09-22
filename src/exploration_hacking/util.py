from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import mlflow

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


class MLFlowLogger:
    def __init__(
        self,
        experiment_name: str,
        concurrent: bool = False,
        max_workers: int | None = 100,
    ):
        try:
            mlflow.create_experiment(experiment_name)
        except Exception as e:
            print(f"Error creating MLFlow experiment: {e}")
        mlflow.set_experiment(experiment_name)

        self.concurrent = concurrent
        self.max_workers = max_workers

    def _log_one_span(self, inputs, outputs, tags, name="generation"):
        span = mlflow.start_span_no_context(name, inputs=inputs, tags=tags)
        try:
            span.set_outputs(outputs)
        finally:
            span.end()

    def log_spans(self, all_inputs, all_outputs, all_tags, name="generation"):
        with true_random_context():
            if self.concurrent:
                with ThreadPoolExecutor(
                    max_workers=self.max_workers or len(all_inputs)
                ) as executor:
                    executor.map(
                        self._log_one_span, all_inputs, all_outputs, all_tags, name
                    )
            else:
                for input, output, tag in zip(all_inputs, all_outputs, all_tags):
                    self._log_one_span(input, output, tag, name)

    def log_spans_from_results(self, prompts, completions, rewards, metrics, answers):
        all_inputs = [{"prompt": prompt} for prompt in prompts]
        all_outputs = [
            {"completion": completion, "answer": answer}
            for completion, answer in zip(completions, answers)
        ]
        all_tags = [
            {k: str(v) for k, v in metrics.items()} | {"reward": str(reward)}
            for reward in rewards
        ]
        self.log_spans(all_inputs, all_outputs, all_tags)
