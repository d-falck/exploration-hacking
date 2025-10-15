from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import mlflow
import wandb

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
        # Try to create experiment with original name
        final_experiment_name = experiment_name
        try:
            mlflow.create_experiment(experiment_name)
        except Exception as e:
            error_msg = str(e)
            # Check for MLflow's RESOURCE_ALREADY_EXISTS error
            if (
                "RESOURCE_ALREADY_EXISTS" in error_msg
                or "already exists" in error_msg.lower()
            ):
                counter = 1
                while counter < 100:  # Safety limit
                    final_experiment_name = f"{experiment_name}_{counter}"
                    try:
                        mlflow.create_experiment(final_experiment_name)
                        print(
                            f"Experiment '{experiment_name}' already exists. Created: {final_experiment_name}"
                        )
                        break
                    except Exception as e2:
                        if (
                            "RESOURCE_ALREADY_EXISTS" in str(e2)
                            or "already exists" in str(e2).lower()
                        ):
                            counter += 1
                            continue
                        else:
                            print(f"Error creating MLFlow experiment: {e2}")
                            final_experiment_name = (
                                experiment_name  # Fall back to original
                            )
                            break
            else:
                print(f"Error creating MLFlow experiment: {e}")

        mlflow.set_experiment(final_experiment_name)

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
                names = [name] * len(all_inputs)
                with ThreadPoolExecutor(
                    max_workers=self.max_workers or len(all_inputs)
                ) as executor:
                    executor.map(
                        self._log_one_span, all_inputs, all_outputs, all_tags, names
                    )
            else:
                for input, output, tag in zip(all_inputs, all_outputs, all_tags):
                    self._log_one_span(input, output, tag, name)

    def log_spans_from_results(
        self,
        prompts,
        completions,
        rewards: list[float] | None = None,
        metrics: dict[str, list[float]] | None = None,
        answers: list[str] | None = None,
        infos: list[dict] | None = None,
        **extra_tags: dict,
    ):
        all_inputs = [{"prompt": prompt} for prompt in prompts]
        all_outputs = []

        for i, completion in enumerate(completions):
            output = {"completion": completion}

            if answers:
                output["answer"] = answers[i]

            if infos:
                judge_response = infos[i].get("judge_response")
                if judge_response:
                    output["judge_response"] = judge_response

            all_outputs.append(output)

        all_tags = []
        for i in range(len(all_outputs)):
            tags = {k: str(v[i]) for k, v in metrics.items()}
            tags |= extra_tags

            if rewards:
                tags["reward"] = str(rewards[i])

            if wandb.run:
                tags["wandb_run_id"] = wandb.run.id

            if infos:
                segment = infos[i].get("segment")
                if segment:
                    tags["segment"] = segment

            all_tags.append(tags)

        self.log_spans(all_inputs, all_outputs, all_tags)
