# GRPO config

set -x
set -a && source .env && set +a

python -m verl.trainer.main_ppo \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=grpo \
    data.train_files=data/verl/train.parquet \
    data.val_files=data/verl/eval.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="/workspace/models/Qwen3-14B" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='[console,wandb]' \
    trainer.project_name=science-grpo-verl \
    trainer.experiment_name=verl-grpo-test \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=0 \
    trainer.total_epochs=10 \
    custom_reward_function.path=src/exploration_hacking/rewards/verl.py

# Notes:
# - algorithm.use_kl_in_reward=False: GRPO uses KL in loss, not reward
# - data.train_batch_size: Number of prompts (not rollouts) per training epoch
# - actor.ppo_mini_batch_size: Number of prompts (not rollouts) per gradient update
# - actor.ppo_micro_batch_size_per_gpu: How many samples fit on a single GPU for a gradient update
# - actor.ppo_epochs: Number of times to repeat training on the same data before generating new rollouts
# - actor.use_kl_loss=True: GRPO uses KL in loss, not reward
# - rollout.n: Number of rollouts per prompt
# - rollout.log_prob_micro_batch_size_per_gpu: How many rollouts fit on a single GPU for log probability computation
# - rollout.tensor_model_parallel_size: For multi-GPU vLLM generation (number of GPUs)
# - rollout.load_format=safetensors: Needed for LoRA
#
# TODOs:
# - experiment_name: change for your experiment
# - target_modules: maybe change from all-linear
# - use_shm: verify if needed
# - layered_summon: maybe enable for large models (see verl LoRA guide)
# - param_offload/optimizer_offload: verify if needed
# - actor.kl_loss_coeff: maybe we need this