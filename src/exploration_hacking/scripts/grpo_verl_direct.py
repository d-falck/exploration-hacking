import random
from pathlib import Path

from dotenv import load_dotenv
import torch
import wandb

from exploration_hacking.config import ExperimentConfig
from exploration_hacking.data import DataConfig, load_dataset
from exploration_hacking.logging import add_completion_logging, save_completion_logs
from exploration_hacking.rewards.factory import RewardConfig, get_reward_functions
from exploration_hacking.rl_verl import RLConfig
from exploration_hacking.model_verl import ModelConfig

load_dotenv()


class Config(ExperimentConfig):
    data: DataConfig
    model: ModelConfig
    reward: RewardConfig
    rl: RLConfig


def main(config: Config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name or "verl-grpo",
        config=dict(config=config.model_dump()),
    )
    
    run_name = wandb.run.name
    output_dir = Path(config.rl.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset, problems = load_dataset(config.data)
    
    # Save dataset as parquet
    data_path = output_dir / "train_data.parquet"
    dataset.to_parquet(data_path)
    
    # Setup rewards
    reward_funcs = get_reward_functions(
        config.data.dataset_name, problems, config.reward, None
    )
    reward_funcs = [add_completion_logging(f) for f in reward_funcs]
    
    # Create reward function for VERL
    def compute_rewards(prompts, completions, **kwargs):
        import asyncio
        task_ids = kwargs.get('task_id', [])
        rewards = []
        
        for task_id, prompt, completion in zip(task_ids, prompts, completions):
            if task_id in problems:
                problem = problems[task_id]
                reward_values = []
                
                for func in reward_funcs:
                    if asyncio.iscoroutinefunction(func):
                        r = asyncio.run(func(problem, completion))
                    else:
                        r = func(problem, completion)
                    
                    reward_values.append(
                        r.get('reward', 0) if isinstance(r, dict) else float(r)
                    )
                
                rewards.append(sum(reward_values))
            else:
                rewards.append(0.0)
        
        return rewards
    
    try:
        from verl import PPOTrainer
        from verl.config import PPOConfig
        
        # Create VERL config
        verl_config = PPOConfig(
            algorithm_type="grpo",
            model_name=config.model.model_name,
            train_dataset_path=str(data_path),
            output_dir=str(output_dir),
            
            # Training params
            num_train_epochs=config.rl.num_epochs,
            train_batch_size=config.rl.batch_size * config.rl.num_rollouts,
            learning_rate=config.rl.learning_rate,
            warmup_ratio=config.rl.warmup_ratio,
            weight_decay=config.rl.weight_decay,
            gradient_accumulation_steps=config.rl.gradient_accumulation_steps,
            
            # GRPO params
            num_rollouts_per_prompt=config.rl.num_rollouts,
            kl_coef=config.rl.kl_loss_coef,
            temperature=config.rl.temperature,
            top_p=config.rl.top_p,
            
            # Model params
            max_seq_length=config.model.max_seq_length,
            lora_rank=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
        )
        
        # Create trainer
        trainer = PPOTrainer(
            config=verl_config,
            reward_function=compute_rewards,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir / "final")
        
    except ImportError:
        print("VERL not installed. Install with: pip install verl[vllm]")
    finally:
        save_completion_logs(str(output_dir))
        wandb.finish()


if __name__ == "__main__":
    main(Config())