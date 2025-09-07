# run on 8xH100
# make sure your current working directory is the root of the project

set -a && source /workspace/exploration-hacking/.env && set +a

set -x

ulimit -n 65535

PROJECT_DIR="/workspace/exploration-hacking"
CONFIG_PATH="/workspace/exploration-hacking/etc"

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/etc/tool_config.yaml" \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.return_raw_chat=True \
    data.train_batch_size=256 \
    data.train_files=/workspace/data/gsm8k/train.parquet \
    data.truncation='error' \
    data.val_files=/workspace/data/gsm8k/test.parquet \
    trainer.critic_warmup=0 \
    trainer.experiment_name='qwen3-4b_function_rm-gsm8k-sgl-multi-w-tool-verify-n16' \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.project_name='gsm8k_async_rl' \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    --config-name='gsm8k_multiturn_grpo' \
    --config-path="$CONFIG_PATH" $@

python3 -m verl.trainer.main_ppo \
    --config-name=search_multiturn_grpo \
    --config-path=/home/wzhang784/verl/examples/sglang_multiturn/config \
    'trainer.logger=["console","wandb"]' \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.path=/home/wzhang784/verl/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/home/wzhang784/verl/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.return_raw_chat=True \
    data.tokenizer=/home/wzhang784/verl/Qwen2.5-3B-Instruct \
    data.train_batch_size=64 \
    data.train_files=/home/wzhang784/Search-R1/data/musique_train_test/train.parquet \
    data.truncation=error \
    data.val_batch_size=128 \
    data.val_files=/home/wzhang784/Search-R1/data/musique_train_test/train.parquet \
    trainer.critic_warmup=0 \
    trainer.experiment_name=qwen2.5-3b-instruct_function_rm-search-async-sgl-multi-w-searchtool-verify-n16 \
    trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.project_name=search_r1_like_async_rl \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \