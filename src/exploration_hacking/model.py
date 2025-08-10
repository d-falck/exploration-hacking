from pydantic import BaseModel
from unsloth import FastLanguageModel


class ModelConfig(BaseModel):
    model_name: str
    lora_rank: int
    lora_alpha: int
    gpu_memory_utilization: float
    max_seq_length: int


_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_peft_model(config: ModelConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        max_lora_rank=config.lora_rank,
        load_in_4bit=True,
        gpu_memory_utilization=config.gpu_memory_utilization,
        fast_inference=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=_TARGET_MODULES,
        lora_alpha=config.lora_alpha,
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer