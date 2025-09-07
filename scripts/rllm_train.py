from dotenv import load_dotenv

load_dotenv()


from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry

from rllm.agents.system_prompts import SEARCH_SYSTEM_PROMPT
from rllm.agents.tool_agent import ToolAgent
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.tools.web_tools import TavilySearchTool
from rllm.trainer.agent_trainer import AgentTrainer


def prepare_hotpotqa_data(train_size=None, test_size=None):
    """
    Loading HotpotQA dataset and registering it with the DatasetRegistry.
    Only loads essential fields: question, ground_truth, data_source

    Args:
        train_size: Maximum number of training examples to load
        test_size: Maximum number of test examples to load

    Returns:
        tuple: (train_dataset, test_dataset)
    """

    def process_split(split_data, max_size):
        """Process a data split with optional size limit"""
        if max_size is not None:
            split_data = split_data.select(range(min(max_size, len(split_data))))
        print(split_data)
        processed = [{"question": example["question"], "ground_truth": example["answer"], "data_source": "hotpotqa"} for example in split_data]

        print(f"Processed {len(processed)} examples")
        return processed

    print("Loading HotpotQA dataset...")
    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)

    train_processed = process_split(hotpot_dataset["train"], train_size)
    test_processed = process_split(hotpot_dataset["validation"], test_size)

    train_dataset = DatasetRegistry.register_dataset("hotpotqa", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("hotpotqa", test_processed, "test")

    return train_dataset, test_dataset


# override_dict = {
#     "actor_rollout_ref.actor.clip_ratio_high": 0.28,
#     "actor_rollout_ref.actor.entropy_coeff": 0,
#     "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
#     "actor_rollout_ref.actor.fsdp_config.param_offload": True,
#     "actor_rollout_ref.actor.kl_loss_coef": 0.001,
#     "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
#     "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum",
#     "actor_rollout_ref.actor.optim.lr": 1e-6,
#     "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 2048,
#     "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
#     "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
#     "actor_rollout_ref.actor.use_dynamic_bsz": True,
#     "actor_rollout_ref.actor.use_kl_loss": False,
#     "actor_rollout_ref.hybrid_engine": True,
#     "actor_rollout_ref.model.enable_gradient_checkpointing": True,
#     "actor_rollout_ref.model.path": "Qwen/Qwen3-4B",
#     "actor_rollout_ref.model.use_remove_padding": True,
#     "actor_rollout_ref.model.lora_rank": 32,
#     "actor_rollout_ref.model.lora_alpha": 32,
#     "actor_rollout_ref.model.target_modules": "all-linear",
#     "actor_rollout_ref.ref.fsdp_config.param_offload": True,
#     "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
#     "actor_rollout_ref.rollout.chat_scheduler": "verl.schedulers.completions_scheduler.CompletionsScheduler",
#     "actor_rollout_ref.rollout.enforce_eager": False,
#     "actor_rollout_ref.rollout.gpu_memory_utilization": 0.4,
#     "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
#     "actor_rollout_ref.rollout.mode": "async",
#     "actor_rollout_ref.rollout.n": 4,
#     "actor_rollout_ref.rollout.name": "vllm",
#     "actor_rollout_ref.rollout.temperature": 0.7,
#     "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
#     "actor_rollout_ref.rollout.val_kwargs.n": 1,
#     "actor_rollout_ref.rollout.val_kwargs.temperature": 0.7,
#     "actor_rollout_ref.rollout.val_kwargs.top_k": 20,
#     "actor_rollout_ref.rollout.val_kwargs.top_p": 0.8,
#     "agent.async_engine": True,
#     "agent.max_steps": 5,
#     "algorithm.adv_estimator": "grpo",
#     "algorithm.clip_advantages": False,
#     "algorithm.kl_ctrl.kl_coef": 0.001,
#     "algorithm.mask_truncated_samples": False,
#     "data.max_prompt_length": 1024,
#     "data.max_response_length": 1024,
#     "data.train_batch_size": 32,
#     "data.val_batch_size": 10,
#     "trainer.critic_warmup": 0,
#     "trainer.default_hdfs_dir": None,
#     "trainer.experiment_name": "7b-loop-drgrpo-search_agent",
#     "trainer.logger": ["console", "wandb"],
#     "trainer.n_gpus_per_node": 1,
#     "trainer.nnodes": 1,
#     "trainer.project_name": "deepscaler-agent",
#     "trainer.save_freq": 40,
#     "trainer.test_freq": 10,
#     "trainer.total_epochs": 100,
#     "trainer.val_before_train": False,
# }

override_dict = {
    "actor_rollout_ref.actor.clip_ratio_high": 0.28,
    "actor_rollout_ref.actor.entropy_coeff": 0.001,
    "actor_rollout_ref.actor.fsdp_config.fsdp_size": -1,  # From verl_test
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
    "actor_rollout_ref.actor.fsdp_config.param_offload": True,
    "actor_rollout_ref.actor.kl_loss_coef": 0.001,
    "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
    "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-sum",  # Missing in verl_test
    "actor_rollout_ref.actor.optim.lr": 3e-5,
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 2048,  # Missing in verl_test
    "actor_rollout_ref.actor.ppo_mini_batch_size": 8,
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 8,
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
    "actor_rollout_ref.actor.use_dynamic_bsz": True,  # Missing in verl_test
    "actor_rollout_ref.actor.use_kl_loss": True,  # New
    "actor_rollout_ref.hybrid_engine": True,  # Missing in verl_test
    "actor_rollout_ref.model.enable_gradient_checkpointing": True,
    "actor_rollout_ref.model.path": "Qwen/Qwen3-4B",
    "actor_rollout_ref.model.use_remove_padding": True,
    "actor_rollout_ref.model.lora_rank": 32,
    "actor_rollout_ref.model.lora_alpha": 32,
    "actor_rollout_ref.model.target_modules": "all-linear",
    "actor_rollout_ref.ref.fsdp_config.param_offload": True,
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 8,
    "actor_rollout_ref.rollout.chat_scheduler": "verl.schedulers.completions_scheduler.CompletionsScheduler",  # Missing in verl_test
    "actor_rollout_ref.rollout.enable_chunked_prefill": False,  # New
    "actor_rollout_ref.rollout.enforce_eager": False,  # Missing in verl_test
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.4,
    "actor_rollout_ref.rollout.load_format": "safetensors",  # New
    "actor_rollout_ref.rollout.layered_summon": True,  # New
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 8,
    "actor_rollout_ref.rollout.mode": "async",  # New
    "actor_rollout_ref.rollout.max_num_batched_tokens": 1536,  # New
    "actor_rollout_ref.rollout.max_model_len": 1536,  # New
    "actor_rollout_ref.rollout.max_num_seqs": 512,  # New
    "actor_rollout_ref.rollout.n": 5,
    "actor_rollout_ref.rollout.name": "vllm",
    "actor_rollout_ref.rollout.temperature": 0.7,
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.val_kwargs.n": 1,  # Missing in verl_test
    "actor_rollout_ref.rollout.val_kwargs.temperature": 0.7,  # Missing in verl_test
    "actor_rollout_ref.rollout.val_kwargs.top_k": 20,  # Missing in verl_test
    "actor_rollout_ref.rollout.val_kwargs.top_p": 0.8,  # Missing in verl_test
    "agent.async_engine": True,
    "agent.max_steps": 5,
    "algorithm.adv_estimator": "grpo",
    "algorithm.clip_advantages": False,
    "algorithm.kl_ctrl.kl_coef": 0.001,
    "algorithm.use_kl_in_reward": False,  # New
    "algorithm.mask_truncated_samples": False,  # Missing in verl_test
    "data.max_prompt_length": 512,
    "data.max_response_length": 1024,
    "data.train_batch_size": 8,
    "data.val_batch_size": 8,
    "trainer.critic_warmup": 0,
    "trainer.default_hdfs_dir": None,  # Missing in verl_test
    "trainer.experiment_name": "7b-loop-drgrpo-search_agent",
    "trainer.logger": ["console", "wandb"],
    "trainer.n_gpus_per_node": 1,
    "trainer.nnodes": 1,
    "trainer.project_name": "deepscaler-agent",
    "trainer.save_freq": 40,
    "trainer.test_freq": 10,
    "trainer.total_epochs": 1,
    "trainer.val_before_train": False,
}

render = lambda x: "null" if x is None else x
overrides = [f"++{k}={render(v)}" for k, v in override_dict.items()]

from hydra import compose, initialize

with initialize(config_path="../rllm/rllm/trainer/config", version_base=None):
    config = compose(config_name="agent_ppo_trainer", overrides=overrides)

import os
import rllm
rllm_dir = os.path.dirname(os.path.dirname(rllm.__file__))

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"  # Was true but errored
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "100000000000"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["RLLM_DIR"] = rllm_dir

prepare_hotpotqa_data(train_size=32, test_size=10)

train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
val_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")

tool_map = {"web_search": TavilySearchTool}

env_args = {
    "max_steps": 10,
    "tool_map": tool_map,
    "reward_fn": search_reward_fn,
}

agent_args = {"system_prompt": SEARCH_SYSTEM_PROMPT, "tool_map": tool_map, "parser_name": "qwen"}

trainer = AgentTrainer(
    agent_class=ToolAgent,
    env_class=ToolEnvironment,
    config=config,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    agent_args=agent_args,
    env_args=env_args,
)

trainer.train()