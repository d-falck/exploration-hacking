"""Utility functions for logging, metrics, and experiment management."""

import warnings

# Suppress Pydantic v2 compatibility warnings from dependencies
# Must be set before importing inspect_ai or verifiers
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import mlflow
import wandb
import os
import datetime
from typing import Literal

import random

from inspect_ai.log import (
    EvalLog,
    EvalSample,
    EvalSpec,
    EvalDataset,
    EvalConfig,
    EvalSampleScore,
    EvalResults,
    EvalScore,
    EvalMetric,
    EvalStats,
    write_eval_log,
)
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool
from inspect_ai.tool import ToolCall
import json
import statistics
import re


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


class TraceLogger(ABC):
    """Abstract base class for trace loggers that can be used as context managers."""

    @abstractmethod
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
        """Log evaluation results."""
        pass

    @abstractmethod
    def close(self):
        """Cleanup/finalize logging."""
        pass

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and call close."""
        self.close()
        return False


def configure_mlflow_connection_pool(pool_size: int = 100):
    """
    Configure requests connection pool settings for MLflow.

    This prevents "Connection pool is full" warnings when logging many
    spans concurrently to MLflow.

    Args:
        pool_size: Maximum number of connections in the pool. Should match
                   or exceed the number of concurrent workers.
    """
    # Disabled for now - urllib3 patching was causing errors
    # The connection pool warnings are harmless and can be ignored

    # import urllib3.poolmanager
    # import urllib3.connectionpool
    # from urllib3.util.retry import Retry

    # # Set connection pool size via environment variables
    # os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    # os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "120")

    # # Patch PoolManager initialization to use larger pool sizes
    # if not hasattr(urllib3.poolmanager.PoolManager, '_original_init'):
    #     urllib3.poolmanager.PoolManager._original_init = urllib3.poolmanager.PoolManager.__init__

    # def patched_pool_manager_init(self, *args, **kwargs):
    #     kwargs.setdefault('maxsize', pool_size)
    #     kwargs.setdefault('block', False)
    #     return urllib3.poolmanager.PoolManager._original_init(self, *args, **kwargs)

    # urllib3.poolmanager.PoolManager.__init__ = patched_pool_manager_init

    # # Also patch HTTPConnectionPool to increase individual pool sizes
    # if not hasattr(urllib3.connectionpool.HTTPConnectionPool, '_original_init'):
    #     urllib3.connectionpool.HTTPConnectionPool._original_init = urllib3.connectionpool.HTTPConnectionPool.__init__

    # def patched_connection_pool_init(self, *args, **kwargs):
    #     kwargs.setdefault('maxsize', pool_size)
    #     kwargs.setdefault('block', False)
    #     return urllib3.connectionpool.HTTPConnectionPool._original_init(self, *args, **kwargs)

    # urllib3.connectionpool.HTTPConnectionPool.__init__ = patched_connection_pool_init
    pass


class MLFlowTraceLogger(TraceLogger):
    def __init__(
        self,
        experiment_name: str,
        concurrent: bool = True,
        max_workers: int | None = 100,
        use_process: bool = False,
    ):
        # Configure connection pool to handle concurrent logging
        pool_size = max(max_workers or 100, 100)
        configure_mlflow_connection_pool(pool_size)

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
        self.experiment_name = final_experiment_name

        self.concurrent = concurrent
        self.max_workers = max_workers
        self.use_process = use_process
        
        # Process-based logging setup
        if self.use_process:
            self.queue = mp.Queue()
            self.worker = mp.Process(target=self._worker_loop, daemon=True)
            self.worker.start()

    def _worker_loop(self):
        """Worker process that logs spans from queue."""
        mlflow.set_experiment(self.experiment_name)
        while True:
            item = self.queue.get()
            if item is None:  # Stop signal
                break
            inputs, outputs, tags, name = item
            try:
                self._do_log_span(inputs, outputs, tags, name)
            except Exception as e:
                print(f"Error logging span {name} with tags {tags}: {e}")
    
    def _do_log_span(self, inputs, outputs, tags, name="generation"):
        """Actually log the span to MLflow."""
        print(f"Logging span {name} with tags {tags}")
        span = mlflow.start_span_no_context(name, inputs=inputs, tags=tags)
        try:
            span.set_outputs(outputs)
        finally:
            span.end()
    
    def _log_one_span(self, inputs, outputs, tags, name="generation"):
        if self.use_process:
            self.queue.put((inputs, outputs, tags, name))
        else:
            self._do_log_span(inputs, outputs, tags, name)

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
                # Include all judge responses (supports multiple judges)
                for key, value in infos[i].items():
                    if key.startswith("judge_response"):
                        output[key] = value

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
    
    def close(self):
        """Stop the background worker if using process mode."""
        if self.use_process:
            self.queue.put(None)  # Stop signal
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                self.worker.terminate()


# Backwards compatibility alias
MLFlowLogger = MLFlowTraceLogger


class InspectTraceLogger(TraceLogger):
    """Logger that accumulates traces and saves them as an Inspect AI .eval file."""

    def __init__(
        self,
        experiment_name: str,
        output_path: str,
        model_name: str,
        task_metadata: dict | None = None,
        compute_summary_stats: bool = True,
        compute_segment_stats: bool = True,
        save_every_n_logs: int | None = 1,
        **kwargs,
    ):
        self.experiment_name = experiment_name
        self.output_path = output_path
        self.model_name = model_name
        self.task_metadata = task_metadata or {}
        self.compute_summary_stats = compute_summary_stats
        self.compute_segment_stats = compute_segment_stats
        self.save_every_n_logs = save_every_n_logs

        # Accumulate data during logging
        self.prompts = []
        self.completions = []
        self.rewards = []
        self.metrics = defaultdict(list)
        self.infos = []
        self.answers = []
        self.tasks = []

        # Track number of log calls for periodic saving
        self.log_call_count = 0

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
        """Accumulate evaluation results for later saving."""
        self.prompts.extend(prompts)
        self.completions.extend(completions)

        if rewards:
            self.rewards.extend(rewards)
        else:
            self.rewards.extend([0.0] * len(prompts))

        if metrics:
            for metric_name, metric_values in metrics.items():
                self.metrics[metric_name].extend(metric_values)

        if answers:
            self.answers.extend(answers)
        else:
            self.answers.extend([""] * len(prompts))

        if infos:
            self.infos.extend(infos)
        else:
            self.infos.extend([{}] * len(prompts))

        # Use experiment_name as task name for each sample
        self.tasks.extend([self.experiment_name] * len(prompts))

        # Increment log call counter and check if we should save
        self.log_call_count += 1
        if self.save_every_n_logs and self.log_call_count % self.save_every_n_logs == 0:
            self._periodic_save()

    def _periodic_save(self):
        """Save current accumulated data to .eval file (checkpoint)."""
        if not self.prompts:
            return

        # Import here to avoid circular dependency
        try:
            from verifiers.types import GenerateOutputs
        except ImportError:
            import sys
            print("Warning: Could not import GenerateOutputs, cannot save .eval file")
            return

        # Create checkpoint filename with iteration number
        base_path = self.output_path.rsplit('.', 1)[0] if '.' in self.output_path else self.output_path
        checkpoint_path = f"{base_path}_checkpoint_{self.log_call_count}.eval"

        # Convert accumulated data to GenerateOutputs
        outputs = GenerateOutputs(
            prompt=self.prompts,
            completion=self.completions,
            answer=self.answers,
            state=[{}] * len(self.prompts),  # Empty states
            info=self.infos,
            task=self.tasks,
            reward=self.rewards,
            metrics=dict(self.metrics),
        )

        # Save using the adapter
        InspectEvalAdapter.save_as_eval(
            outputs=outputs,
            model_name=self.model_name,
            output_path=checkpoint_path,
            task_name=self.experiment_name,
            task_metadata=self.task_metadata,
            compute_summary_stats=self.compute_summary_stats,
            compute_segment_stats=self.compute_segment_stats,
        )

        print(f"Periodic save: {len(self.prompts)} samples to {checkpoint_path} (log iteration {self.log_call_count})")

    def close(self):
        """Convert accumulated data to GenerateOutputs and save as .eval file."""
        if not self.prompts:
            print("No data logged, skipping .eval file creation")
            return

        # Import here to avoid circular dependency
        try:
            from verifiers.types import GenerateOutputs
        except ImportError:
            import sys
            print("Warning: Could not import GenerateOutputs, cannot save .eval file")
            return

        # Convert accumulated data to GenerateOutputs
        outputs = GenerateOutputs(
            prompt=self.prompts,
            completion=self.completions,
            answer=self.answers,
            state=[{}] * len(self.prompts),  # Empty states
            info=self.infos,
            task=self.tasks,
            reward=self.rewards,
            metrics=dict(self.metrics),
        )

        # Save using the adapter
        InspectEvalAdapter.save_as_eval(
            outputs=outputs,
            model_name=self.model_name,
            output_path=self.output_path,
            task_name=self.experiment_name,
            task_metadata=self.task_metadata,
            compute_summary_stats=self.compute_summary_stats,
            compute_segment_stats=self.compute_segment_stats,
        )


class InspectEvalAdapter:
    """Adapter to convert GenerateOutputs to Inspect AI .eval format."""

    @staticmethod
    def _convert_chat_message_to_inspect(message: dict):
        """
        Convert OpenAI ChatCompletionMessageParam to Inspect AI message format.

        Args:
            message: OpenAI format message dict with 'role' and 'content' keys,
                    optionally 'tool_calls', 'tool_call_id'

        Returns:
            Inspect AI ChatMessage (ChatMessageUser, ChatMessageAssistant, ChatMessageSystem, or ChatMessageTool)
        """
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "user":
            return ChatMessageUser(content=content)
        elif role == "assistant":
            # Handle tool calls if present
            tool_calls_openai = message.get("tool_calls")
            tool_calls_inspect = None

            if tool_calls_openai:
                tool_calls_inspect = []
                for tc in tool_calls_openai:
                    # Handle both ChatCompletionMessageToolCall objects and dicts
                    if hasattr(tc, 'id'):
                        # It's a ChatCompletionMessageToolCall object
                        tc_id = tc.id
                        tc_function = tc.function.name
                        tc_arguments = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    else:
                        # It's a dict
                        tc_id = tc.get("id", "")
                        tc_function = tc.get("function", {}).get("name", "")
                        tc_args = tc.get("function", {}).get("arguments", "{}")
                        tc_arguments = json.loads(tc_args) if isinstance(tc_args, str) else tc_args

                    # Double-check: if tc_arguments is still a string after parsing, parse it again
                    # This handles cases where the data was double-JSON-encoded
                    if isinstance(tc_arguments, str):
                        try:
                            tc_arguments = json.loads(tc_arguments)
                        except (json.JSONDecodeError, TypeError):
                            # If it can't be parsed, wrap it in a dict
                            tc_arguments = {"raw": tc_arguments}

                    tool_calls_inspect.append(
                        ToolCall(
                            id=tc_id,
                            function=tc_function,
                            arguments=tc_arguments,
                            type="function"
                        )
                    )

            return ChatMessageAssistant(content=content, tool_calls=tool_calls_inspect)
        elif role == "tool":
            # Tool response message
            tool_call_id = message.get("tool_call_id")
            function = message.get("function")  # Optional in OpenAI format
            return ChatMessageTool(
                content=content,
                tool_call_id=tool_call_id,
                function=function
            )
        elif role == "system":
            return ChatMessageSystem(content=content)
        else:
            # Default to user message for unknown roles
            return ChatMessageUser(content=content)

    @staticmethod
    def save_as_eval(
        outputs,  # GenerateOutputs type (avoiding import to prevent circular dependency)
        model_name: str,
        output_path: str,
        task_name: str = "evaluation",
        task_metadata: dict | None = None,
        compute_summary_stats: bool = True,
        compute_segment_stats: bool = True,
    ):
        """
        Convert GenerateOutputs to Inspect AI EvalLog format and save as .eval file.

        Args:
            outputs: GenerateOutputs object containing prompts, completions, answers, rewards, metrics
            model_name: Name of the model used for generation
            output_path: Path where the .eval file should be saved
            task_name: Name of the evaluation task (default: "evaluation")
            task_metadata: Optional metadata dict for task-level information
            compute_summary_stats: If True, compute and add summary statistics (default: True)
            compute_segment_stats: If True, compute segment-level statistics when segments are present (default: True)

        Example:
            >>> from verifiers import GenerateOutputs
            >>> outputs = GenerateOutputs(...)
            >>> InspectEvalAdapter.save_as_eval(
            ...     outputs=outputs,
            ...     model_name="gpt-4",
            ...     output_path="results.eval",
            ...     task_name="my_task",
            ...     task_metadata={"domain": "math", "difficulty": "hard"},
            ...     compute_summary_stats=True
            ... )
        """
        samples = []

        # Create one EvalSample per index
        for i in range(len(outputs.prompt)):
            # Convert prompt messages
            prompt_messages = outputs.prompt[i]
            if isinstance(prompt_messages, list):
                messages = [
                    InspectEvalAdapter._convert_chat_message_to_inspect(msg)
                    for msg in prompt_messages
                ]
            else:
                # If it's a string, wrap it in a user message
                messages = [ChatMessageUser(content=str(prompt_messages))]

            # Convert completion messages and append to messages list
            completion_messages = outputs.completion[i]
            if isinstance(completion_messages, list):
                messages.extend([
                    InspectEvalAdapter._convert_chat_message_to_inspect(msg)
                    for msg in completion_messages
                ])
            else:
                # If it's a string, wrap it in an assistant message
                messages.append(ChatMessageAssistant(content=str(completion_messages)))

            # Extract input (problem text) from prompt messages
            # Use just the first user message as a string for input
            input_text = ""
            if isinstance(prompt_messages, list):
                # Find the first user message
                for msg in prompt_messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        input_text = content if isinstance(content, str) else str(content)
                        break
                # If no user message found, use the first message's content
                if not input_text and prompt_messages:
                    first_msg = prompt_messages[0]
                    content = first_msg.get("content", "")
                    input_text = content if isinstance(content, str) else str(content)
            else:
                # If it's a string, use it directly
                input_text = str(prompt_messages)

            # Extract info for this sample (used for judge responses and metadata)
            sample_info = {}
            if outputs.info and i < len(outputs.info):
                sample_info = outputs.info[i] if outputs.info[i] else {}

            # Build scores dict from reward and metrics
            # Include judge responses as explanations when available

            # First, collect all scores (reward + metrics)
            all_scores = {}
            if outputs.reward and i < len(outputs.reward):
                all_scores["reward"] = outputs.reward[i]

            for metric_name, metric_values in outputs.metrics.items():
                if i < len(metric_values):
                    all_scores[metric_name] = metric_values[i]

            # Map judge responses to their corresponding scores
            # Strategy: if judge_response has a score tag, match it to the metric with that value
            judge_explanations = {}

            # Check for named judge responses (e.g., judge_response_accuracy)
            for key in sample_info.keys():
                if key.startswith("judge_response_"):
                    metric_name = key.replace("judge_response_", "")
                    # Try direct match first
                    if metric_name in all_scores:
                        judge_explanations[metric_name] = sample_info[key]
                    else:
                        # Try to match with segment prefix (e.g., eval_segment_eval_use_tool_badly)
                        # Look for any score key that ends with _segment_{metric_name}
                        for score_key in all_scores.keys():
                            if score_key.endswith(f"_segment_{metric_name}"):
                                judge_explanations[score_key] = sample_info[key]
                                break

            # Handle generic "judge_response" by matching the score
            if "judge_response" in sample_info and "judge_response" not in judge_explanations:
                judge_text = sample_info["judge_response"]
                # Extract score from judge response
                score_match = re.search(r'<score>([\d.]+)</score>', judge_text)
                if score_match:
                    judge_score = float(score_match.group(1))
                    # Find which metric this score matches
                    for score_name, score_value in all_scores.items():
                        if abs(score_value - judge_score) < 0.01:  # Allow small floating point difference
                            judge_explanations[score_name] = judge_text
                            break
                else:
                    # If no score tag found, default to reward
                    judge_explanations["reward"] = judge_text

            # Build the scores dict with explanations
            scores = {}
            if outputs.reward and i < len(outputs.reward):
                scores["reward"] = EvalSampleScore(
                    value=outputs.reward[i],
                    explanation=judge_explanations.get("reward")
                )

            for metric_name, metric_values in outputs.metrics.items():
                if i < len(metric_values):
                    scores[metric_name] = EvalSampleScore(
                        value=metric_values[i],
                        explanation=judge_explanations.get(metric_name)
                    )

            # Add segment indicator scores if segment info is present
            if sample_info and "segment" in sample_info:
                segment_name = sample_info["segment"]
                # Add a score indicating which segment this sample belongs to
                scores[f"segment_{segment_name}"] = EvalSampleScore(
                    value=1.0,
                    explanation=f"Sample belongs to segment: {segment_name}"
                )

            # Extract metadata from info (excluding judge responses which are now in scores)
            sample_metadata = {}
            if sample_info:
                sample_metadata = {
                    k: v for k, v in sample_info.items()
                    if not k.startswith("judge_response")
                }

            # Create the sample
            sample_kwargs = {
                "id": i,
                "epoch": 1,
                "input": input_text,  # Simple string with the user's question/problem
                "target": outputs.answer[i] if i < len(outputs.answer) else "",
                "messages": messages,  # Full conversation including prompt and completion
            }

            if scores:
                sample_kwargs["scores"] = scores
            if sample_metadata:
                sample_kwargs["metadata"] = sample_metadata

            sample = EvalSample(**sample_kwargs)

            samples.append(sample)

        # Compute summary statistics if requested
        eval_results = None
        if compute_summary_stats:
            eval_scores = []

            # Identify segments first if computing segment stats
            segments = set()
            if compute_segment_stats:
                for info in outputs.info:
                    if info and "segment" in info:
                        segments.add(info["segment"])

            # First, compute overall statistics
            # Compute statistics for reward
            if outputs.reward:
                reward_values = [r for r in outputs.reward if r is not None]
                if reward_values:
                    reward_metrics = {
                        "mean": EvalMetric(name="mean", value=statistics.mean(reward_values)),
                        "median": EvalMetric(name="median", value=statistics.median(reward_values)),
                    }
                    if len(reward_values) > 1:
                        reward_metrics["std"] = EvalMetric(name="std", value=statistics.stdev(reward_values))

                    # Add segment-level statistics for reward
                    if compute_segment_stats and segments:
                        for segment in sorted(segments):
                            segment_indices = [
                                i for i, info in enumerate(outputs.info)
                                if info and info.get("segment") == segment
                            ]
                            segment_reward_values = [
                                outputs.reward[i] for i in segment_indices
                                if i < len(outputs.reward) and outputs.reward[i] is not None
                            ]
                            if segment_reward_values:
                                reward_metrics[f"segment_mean_{segment}"] = EvalMetric(
                                    name=f"segment_mean_{segment}",
                                    value=statistics.mean(segment_reward_values)
                                )
                                reward_metrics[f"segment_median_{segment}"] = EvalMetric(
                                    name=f"segment_median_{segment}",
                                    value=statistics.median(segment_reward_values)
                                )
                                if len(segment_reward_values) > 1:
                                    reward_metrics[f"segment_std_{segment}"] = EvalMetric(
                                        name=f"segment_std_{segment}",
                                        value=statistics.stdev(segment_reward_values)
                                    )

                    eval_scores.append(EvalScore(
                        name="reward",
                        scorer="reward",
                        metrics=reward_metrics,
                        scored_samples=len(reward_values),
                        unscored_samples=len(outputs.reward) - len(reward_values),
                    ))

            # Compute statistics for each metric
            for metric_name, metric_values in outputs.metrics.items():
                values = [v for v in metric_values if v is not None]
                if values:
                    metric_stats = {
                        "mean": EvalMetric(name="mean", value=statistics.mean(values)),
                        "median": EvalMetric(name="median", value=statistics.median(values)),
                    }
                    if len(values) > 1:
                        metric_stats["std"] = EvalMetric(name="std", value=statistics.stdev(values))

                    # Compute segment-level statistics for each metric
                    if compute_segment_stats and segments:
                        for segment in sorted(segments):
                            segment_indices = [
                                i for i, info in enumerate(outputs.info)
                                if info and info.get("segment") == segment
                            ]
                            segment_metric_values = [
                                metric_values[i] for i in segment_indices
                                if i < len(metric_values) and metric_values[i] is not None
                            ]
                            if segment_metric_values:
                                metric_stats[f"segment_mean_{segment}"] = EvalMetric(
                                    name=f"segment_mean_{segment}",
                                    value=statistics.mean(segment_metric_values)
                                )
                                metric_stats[f"segment_median_{segment}"] = EvalMetric(
                                    name=f"segment_median_{segment}",
                                    value=statistics.median(segment_metric_values)
                                )
                                if len(segment_metric_values) > 1:
                                    metric_stats[f"segment_std_{segment}"] = EvalMetric(
                                        name=f"segment_std_{segment}",
                                        value=statistics.stdev(segment_metric_values)
                                    )

                    eval_scores.append(EvalScore(
                        name=metric_name,
                        scorer=metric_name,
                        metrics=metric_stats,
                        scored_samples=len(values),
                        unscored_samples=len(metric_values) - len(values),
                    ))

            eval_results = EvalResults(
                total_samples=len(samples),
                completed_samples=len(samples),
                scores=eval_scores,
            )

        # Create timestamps for stats
        now = datetime.datetime.now(datetime.UTC).isoformat()
        eval_stats = EvalStats(
            started_at=now,
            completed_at=now,
            model_usage={},
        )

        # Create EvalLog
        eval_log = EvalLog(
            status="success",
            eval=EvalSpec(
                created=now,
                task=task_name,
                dataset=EvalDataset(),
                model=model_name,
                config=EvalConfig(),
                metadata=task_metadata,
            ),
            samples=samples,
            results=eval_results,
            stats=eval_stats,
        )

        # Write to file
        write_eval_log(eval_log, output_path)
        print(f"Saved {len(samples)} samples to {output_path}")



def create_trace_logger(
    logger_type: Literal["mlflow", "inspect"],
    experiment_name: str,
    model_name: str | None = None,
    output_path: str | None = None,
    **kwargs
) -> TraceLogger:
    """
    Create a trace logger.

    Args:
        logger_type: Type of logger ("mlflow" or "inspect")
        experiment_name: Name for the experiment/task (used by both)
        model_name: Name of the model (required for inspect, optional for mlflow)
        output_path: Path for output file (required for inspect, ignored by mlflow)
        **kwargs: Additional logger-specific parameters

    Returns:
        TraceLogger: Configured logger instance

    Example:
        >>> # MLFlow logger
        >>> with create_trace_logger("mlflow", "my-experiment") as logger:
        ...     logger.log_spans_from_results(prompts, completions, ...)
        >>> 
        >>> # Inspect logger
        >>> with create_trace_logger("inspect", "my-experiment", 
        ...                          model_name="gpt-4", 
        ...                          output_path="results.eval") as logger:
        ...     logger.log_spans_from_results(prompts, completions, ...)
    """
    if logger_type == "mlflow":
        return MLFlowTraceLogger(
            experiment_name=experiment_name,
            **kwargs,
        )
    elif logger_type == "inspect":
        if not output_path:
            raise ValueError("output_path is required for inspect logger")
        if not model_name:
            raise ValueError("model_name is required for inspect logger")
        return InspectTraceLogger(
            experiment_name=experiment_name,
            output_path=output_path,
            model_name=model_name,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")

