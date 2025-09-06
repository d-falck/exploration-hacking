from pydantic import BaseModel
import verifiers as vf

from exploration_hacking.util import get_batch_params


class _PeftConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] | str = "all-linear"


class _HyperparametersConfig(BaseModel):
    learning_rate: float = 3e-5
    warmup_steps: int = 10
    max_steps: int = 1000
    num_iterations: int = 1
    save_steps: int = 100
    beta: float = 0.001


class _SamplingConfig(BaseModel):
    max_prompt_length: int = 1024
    max_completion_length: int = 2048
    max_seq_len: int = 3072
    max_tokens: int = 1024  # Per response
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = 20


class _BatchingConfig(BaseModel):
    """
    These are reparameterized from the TRL ones to be less confusing.
    """

    mini_batch_size: int = 4
    group_size: int = 4
    rollouts_per_gpu: int = 1


class _EvaluationConfig(BaseModel):
    eval_strategy: str = "steps"
    eval_steps: int = 100
    per_device_eval_batch_size: int = 1  # TODO: increase?


class RLConfig(BaseModel):
    model: str
    peft: _PeftConfig
    hyperparameters: _HyperparametersConfig
    sampling: _SamplingConfig
    batching: _BatchingConfig
    evaluation: _EvaluationConfig


def run_grpo(env: vf.Environment, config: RLConfig, run_name: str):
    model, tokenizer = vf.get_model_and_tokenizer(config.model)

    for k, v in get_batch_params(
        **dict(config.batching),
        num_gpus=len(config.training_gpus),
    ).items():
        setattr(args, k, v)

    args = vf.grpo_defaults(run_name=run_name)
    for field in RLConfig.model_fields:
        if field in ["model", "peft", "batching"]:
            continue

        obj = getattr(config, field)
        cls = obj.__class__
        for field in cls.model_fields:
            setattr(args, field, getattr(obj, field))

    peft_config = vf.lora_defaults(
        r=config.peft.r,
        alpha=config.peft.lora_alpha,
    )
    for field in _PeftConfig.model_fields:
        setattr(peft_config, field, getattr(config.peft, field))

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=args,
        peft_config=peft_config,
    )
    trainer.train()
