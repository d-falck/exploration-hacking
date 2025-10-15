from concurrent.futures import ThreadPoolExecutor
import random
from contextlib import contextmanager

from pydantic import BaseModel, Field
import verifiers as vf
from peft import PeftModel
import wandb

from exploration_hacking.util import get_batch_params


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
    log_to_mlflow: bool = False


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


@contextmanager
def _true_random_context():
    """Temporarily use true randomness, then restore previous state."""
    saved_state = random.getstate()
    try:
        random.seed()  # Reseed with current time
        yield
    finally:
        random.setstate(saved_state)


def _log_traces_to_mlflow(
    all_prompts, all_completions, all_reward_dict, all_states, global_step
):
    print(f"Logging {len(all_prompts)} traces to MLFlow..")
    import mlflow

    with _true_random_context():

        def log_generation(prompt, completion, reward_dict, state):
            inputs = {"prompt": prompt}
            tags = {k: str(v) for k, v in reward_dict.items()}
            tags |= {"step": str(global_step)}
            tags |= {"wandb_run_id": wandb.run.id}

            # Add segment to tags if available
            segment = state.get("info", {}).get("segment")
            if segment:
                tags["segment"] = segment

            span = mlflow.start_span_no_context(
                name="generation", inputs=inputs, tags=tags
            )
            try:
                outputs = {"completion": completion}

                # Add judge response if present
                judge_response = state.get("info", {}).get("judge_response")
                if judge_response:
                    outputs["judge_response"] = judge_response

                span.set_outputs(outputs)
            finally:
                span.end()

        n = len(all_prompts)
        with ThreadPoolExecutor(max_workers=n) as executor:

            for i in range(n):
                prompt = all_prompts[i]
                completion = all_completions[i]
                reward_dict = {k: str(v[i]) for k, v in all_reward_dict.items()}
                state = all_states[i]

                executor.submit(log_generation, prompt, completion, reward_dict, state)
    print(f"MLFlow logging complete.")


def run_grpo(
    env: vf.Environment, config: RLConfig, run_name: str, num_training_gpus: int
):
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
        if field in ["model", "peft", "batching", "shuffle_dataset"]:
            continue

        obj = getattr(config, field)
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
        textual_data_callback=_log_traces_to_mlflow,
    )

    if trainer.accelerator.is_main_process and config.hyperparameters.log_to_mlflow:
        import mlflow

        mlflow.create_experiment(run_name)
        mlflow.set_experiment(run_name)

    trainer.train()
