import json
from pathlib import Path
from collections import defaultdict

from datasets import Dataset
from pydantic import BaseModel
from vllm import SamplingParams
from safetensors import safe_open


class EvalConfig(BaseModel):
    temperature: float
    top_p: float
    num_samples: int
    output_dir: str
    lora_path: str | None = None
    max_tokens: int = 2048


class EvalResult(BaseModel):
    task_ids: list[str]
    prompts: list[list[dict]]
    completions: list[list[str]]
    rewards: dict[str, list[list[float]]]
    mean_rewards: dict[str, float]


def _verify_lora_weights(model_path: str):
    adapter_path = Path(model_path) / "adapter_model.safetensors"
    if not adapter_path.exists():
        print(f"Warning: No adapter weights found at {adapter_path}")
        return

    with safe_open(adapter_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            if n_zeros.item() == 1.0:
                print(f"Warning: Tensor {key} is all zeros")


def _get_inputs(
    dataset: Dataset, tokenizer
) -> tuple[list[str], list[str], list[list[dict]]]:
    inputs = []
    task_ids = []
    prompts = []

    for item in dataset:
        prompt_messages = item["prompt"]
        prompts.append(prompt_messages)

        tokenized = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs.append(tokenized)
        task_ids.append(item["task_id"])

    return inputs, task_ids, prompts


def _extract_completions(all_outputs, inputs):
    completions_by_task = [[] for _ in inputs]

    for outputs in all_outputs:
        for i, output in enumerate(outputs):
            full_text = output.outputs[0].text
            completions_by_task[i].append(full_text.strip())

    return completions_by_task


def _apply_reward_funcs(completions_list, prompts, reward_funcs, task_ids):
    if not reward_funcs:
        return {}, {}

    all_rewards = defaultdict(lambda: defaultdict(list))

    class MockState:
        def __init__(self, step):
            self.global_step = step

    for sample_idx, task_completions in enumerate(zip(*completions_list)):
        formatted_completions = [
            [{"role": "assistant", "content": comp}] for comp in task_completions
        ]

        for func in reward_funcs:
            reward_name = (
                func.__wrapped__.__name__
                if hasattr(func, "__wrapped__")
                else func.__name__
            )

            rewards = func(
                prompts=prompts,
                completions=formatted_completions,
                trainer_state=MockState(sample_idx),
                task_id=task_ids,
            )

            for i, reward in enumerate(rewards):
                all_rewards[reward_name][i].append(float(reward))

    rewards_dict = {}
    mean_rewards = {}

    for reward_name, task_rewards in all_rewards.items():
        rewards_list = [task_rewards[i] for i in range(len(prompts))]
        rewards_dict[reward_name] = rewards_list

        all_values = [r for task in rewards_list for r in task]
        mean_rewards[reward_name] = (
            sum(all_values) / len(all_values) if all_values else 0.0
        )

    return rewards_dict, mean_rewards


def run_eval(
    model,
    tokenizer,
    dataset,
    reward_funcs,
    config: EvalConfig,
) -> EvalResult:

    if config.lora_path:
        _verify_lora_weights(config.lora_path)
        lora_request = model.load_lora(config.lora_path)
    else:
        lora_request = None

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )

    inputs, task_ids, prompts = _get_inputs(dataset, tokenizer)

    print(f"Generating {config.num_samples} samples for {len(inputs)} prompts...")

    all_outputs = []

    for sample_idx in range(config.num_samples):
        print(f"Sample {sample_idx + 1}/{config.num_samples}")

        outputs = model.fast_generate(
            inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        all_outputs.append(outputs)

    completions_list = _extract_completions(all_outputs, inputs)
    rewards_dict, mean_rewards = _apply_reward_funcs(
        completions_list, prompts, reward_funcs, task_ids
    )

    return EvalResult(
        task_ids=task_ids,
        prompts=prompts,
        completions=completions_list,
        rewards=rewards_dict,
        mean_rewards=mean_rewards,
    )


def print_eval_result(result: EvalResult):
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nEvaluated {len(result.task_ids)} tasks")
    print(
        f"Samples per task: {len(result.completions[0]) if result.completions else 0}"
    )

    if result.mean_rewards:
        print("\nMean Rewards:")
        for reward_name, mean_value in result.mean_rewards.items():
            print(f"  {reward_name}: {mean_value:.4f}")

    # Print sample completions
    if result.completions and len(result.completions) > 0:
        print("\nSample completions for first task:")
        first_task = result.task_ids[0]
        print(f"Task: {first_task}")
        for i, completion in enumerate(result.completions[0][:3]):  # Show first 3
            preview = completion[:200] + "..." if len(completion) > 200 else completion
            print(f"\nCompletion {i+1}:\n{preview}")

    print("\n" + "=" * 50)
