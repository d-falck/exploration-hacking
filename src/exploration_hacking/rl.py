from pydantic import BaseModel
import verifiers as vf


class GRPOConfig(BaseModel):
    model: str
    lora_rank: int = 16
    lora_alpha: int = 32


def run_grpo(env: vf.Environment, config: GRPOConfig, run_name: str):
    model, tokenizer = vf.get_model_and_tokenizer(config.model)
    args = vf.grpo_defaults(run_name=run_name)
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=args,
        peft_config=vf.lora_defaults(
            r=config.lora_rank,
            alpha=config.lora_alpha,
        ),
    )
    trainer.train()
