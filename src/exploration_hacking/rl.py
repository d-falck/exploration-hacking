from concurrent.futures import ThreadPoolExecutor
import random
from contextlib import contextmanager
from typing import Literal

from pydantic import BaseModel, Field
import verifiers as vf
from peft import PeftModel
import wandb

from exploration_hacking.util import get_batch_params, MLFlowLogger, create_trace_logger


def _get_model_and_tokenizer_with_lora(
    model_name: str,
    lora_checkpoint: str | None = None,
    use_liger: bool = True,
    model_kwargs: dict | None = None,
    is_trainable: bool = True,
):
    """
    Helper function to load a model and tokenizer, optionally with an existing LoRA checkpoint.

    Args:
        model_name: Base model name/path
        lora_checkpoint: Optional path to existing LoRA checkpoint to load
        use_liger: Whether to use Liger kernels if available
        model_kwargs: Additional kwargs for model loading
        is_trainable: Whether to load LoRA for training (True) or inference (False)

    Returns:
        Tuple of (model, tokenizer) where model may have LoRA loaded
    """
    model, tokenizer = vf.get_model_and_tokenizer(model_name, use_liger, model_kwargs)

    if lora_checkpoint:
        model = PeftModel.from_pretrained(
            model, lora_checkpoint, is_trainable=is_trainable
        )
        print(f"Loaded LoRA checkpoint from: {lora_checkpoint}")
        if is_trainable:
            model.print_trainable_parameters()

    return model, tokenizer


class _PeftConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] | str = "all-linear"
    lora_checkpoint: str | None = None  # Overrides other fields


class _HyperparametersConfig(BaseModel):
    learning_rate: float = 3e-5
    warmup_steps: int = 10
    max_steps: int = 1000
    num_iterations: int = 1
    save_steps: int = 100
    beta: float = 0.001
    max_grad_norm: float = 1000


class _SamplingConfig(BaseModel):
    max_prompt_length: int = 1024
    max_seq_len: int = 3072
    max_tokens: int = 1024  # Per response
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = None


class _BatchingConfig(BaseModel):
    """
    These are reparameterized from the TRL ones to be less confusing.
    """

    mini_batch_size: int = 4
    group_size: int = 4
    rollouts_per_gpu: int = 1
    shuffle_dataset: bool = False  # Disable to preserve our own interleaving


class _EvaluationConfig(BaseModel):
    eval_strategy: str = "steps"
    eval_steps: int = 50000
    per_device_eval_batch_size: int = 1  # TODO: increase?


class RLConfig(BaseModel):
    model: str
    peft: _PeftConfig = Field(default_factory=_PeftConfig)
    hyperparameters: _HyperparametersConfig = Field(
        default_factory=_HyperparametersConfig
    )
    sampling: _SamplingConfig = Field(default_factory=_SamplingConfig)
    batching: _BatchingConfig = Field(default_factory=_BatchingConfig)
    evaluation: _EvaluationConfig = Field(default_factory=_EvaluationConfig)
    logging_destination: Literal["inspect", "mlflow", "none"] = "none"
    logging_output_dir: str | None = None  # Required for inspect logging


@contextmanager
def _true_random_context():
    """Temporarily use true randomness, then restore previous state."""
    saved_state = random.getstate()
    try:
        random.seed()  # Reseed with current time
        yield
    finally:
        random.setstate(saved_state)


_trace_logger = None


def _log_training_traces(
    all_prompts, all_completions, all_reward_dict, all_states, global_step
):
    """Callback to log training traces during GRPO training."""
    global _trace_logger
    if _trace_logger is None:
        return

    print(f"Logging {len(all_prompts)} training traces...")
    _trace_logger.log_spans_from_results(
        all_prompts,
        all_completions,
        metrics=all_reward_dict,
        infos=[state.get("info", {}) for state in all_states],
        step=global_step,
    )
    print(f"Training trace logging complete.")


def run_grpo(
    env: vf.Environment, config: RLConfig, run_name: str, num_training_gpus: int
):
    global _mlflow_logger

    # Use helper function to load model with optional LoRA checkpoint
    model, tokenizer = _get_model_and_tokenizer_with_lora(
        config.model, lora_checkpoint=config.peft.lora_checkpoint, is_trainable=True
    )

    args = vf.grpo_defaults(run_name=run_name)

    # Reparameterize some of the batching config from our less confusing version
    to_reparameterize = ["mini_batch_size", "group_size", "rollouts_per_gpu"]
    for k, v in get_batch_params(
        **config.batching.model_dump(include=to_reparameterize),
        num_gpus=num_training_gpus,
    ).items():
        setattr(args, k, v)
    for k, v in config.batching.model_dump(exclude=to_reparameterize).items():
        setattr(args, k, v)

    for field in RLConfig.model_fields:
        if field in ["model", "peft", "batching", "shuffle_dataset", "log_to_mlflow", "logging_destination", "logging_output_dir"]:
            continue

        obj = getattr(config, field)
        # Only process if it's a Pydantic model with model_fields
        if hasattr(obj.__class__, 'model_fields'):
            cls = obj.__class__
            for field in cls.model_fields:
                setattr(args, field, getattr(obj, field))

    # Handle LoRA checkpoint loading vs creating new LoRA config
    if config.peft.lora_checkpoint:
        # Already loaded LoRA via helper function
        peft_config = None  # No new PEFT config needed
        args.initial_weight_sync = True  # Force initial sync to vLLM
        args.sync_ref_model = False
    else:
        # Create new LoRA config
        peft_config = vf.lora_defaults(
            r=config.peft.r,
            alpha=config.peft.lora_alpha,
        )
        for field in _PeftConfig.model_fields:
            if field != "lora_checkpoint":  # Skip the checkpoint field
                setattr(peft_config, field, getattr(config.peft, field))

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=args,
        peft_config=peft_config,
        textual_data_callback=_log_training_traces,
    )

    # Set up trace logging based on configuration
    global _trace_logger

    # Prepare logger parameters
    should_log = trainer.accelerator.is_main_process and config.logging_destination != "none"

    if should_log and config.logging_destination == "inspect":
        if not config.logging_output_dir:
            raise ValueError("logging_output_dir must be set when using inspect logging")
        import os
        os.makedirs(config.logging_output_dir, exist_ok=True)
        output_path = os.path.join(config.logging_output_dir, f"{run_name}_training.eval")
    else:
        output_path = None

    # Use context manager for logger
    if should_log:
        with create_trace_logger(
            logger_type=config.logging_destination,
            experiment_name=run_name,
            model_name=config.model,
            output_path=output_path,
            task_metadata={"training_run": run_name},
        ) as logger:
            _trace_logger = logger
            trainer.train()
            _trace_logger = None
    else:
        trainer.train()
