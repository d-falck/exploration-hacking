from dotenv import load_dotenv
from pathlib import Path
import logging
import pickle
import random
import os
import subprocess
import sys
import yaml

import numpy as np
import torch
import wandb
from datasets import load_dataset, Dataset, load_from_disk
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from exploration_hacking.dtypes import ExperimentConfig


load_dotenv()
logging.basicConfig(level=logging.INFO)


class _ModelConfig(BaseModel):
    model_name: str
    lora_rank: int = 16
    lora_alpha: int = 32


class _DatasetConfig(BaseModel):
    local_path: Path | None = None
    path: str | None = None
    name: str | None = None
    split: str | None = None


class _TrainingConfig(BaseModel):
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    save_steps: int = 500


class _EvaluationConfig(BaseModel):
    enabled: bool = False
    eval_steps: int = 25
    eval_config_path: str


class Config(ExperimentConfig):
    model: _ModelConfig
    dataset: _DatasetConfig
    training: _TrainingConfig
    evaluation: _EvaluationConfig | None = None
    output_dir: str
    gpus: list[int]
    transform_to_single_turn: bool = True


def transform_multi_turn_to_single_turn(dataset):
    """Transform multi-turn rollouts into separate samples for each assistant completion."""
    new_samples = []

    for sample in dataset:
        prompt = sample["prompt"]
        completion = sample["completion"]
        tools = sample["tools"]

        context = prompt
        for message in completion:
            if message["role"] == "assistant":
                new_sample = {
                    "prompt": context,
                    "completion": [message],
                    "tools": tools,
                }
                new_samples.append(new_sample)

            context = context + [message]

    return Dataset.from_list(new_samples)


class PeriodicEvalCallback(TrainerCallback):
    """Callback to run periodic evaluation during SFT training."""

    def __init__(self, eval_config: _EvaluationConfig, output_dir: str, gpu_id: int):
        self.eval_config = eval_config
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        self.last_eval_step = -1

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Trigger evaluation every eval_steps gradient steps."""
        # Check if we should evaluate (every N gradient steps)
        if (state.global_step > 0 and
            state.global_step % self.eval_config.eval_steps == 0 and
            state.global_step != self.last_eval_step):

            logging.info(f"Triggering evaluation at step {state.global_step}")
            self._run_evaluation(state.global_step, model, args)
            self.last_eval_step = state.global_step

        return control

    def _run_evaluation(self, global_step: int, model, args):
        """Run evaluation subprocess with current checkpoint."""
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        logging.info(f"Saving checkpoint to {checkpoint_path}")
        model.save_pretrained(checkpoint_path)

        # Load eval config
        eval_config_path = Path(self.eval_config.eval_config_path)
        with open(eval_config_path) as f:
            eval_cfg = yaml.safe_load(f)

        # Update checkpoint path in eval config
        eval_cfg['eval']['backend']['lora_path'] = checkpoint_path

        # Create temp eval config file
        temp_eval_path = os.path.join(self.output_dir, f"eval_config_step_{global_step}.yaml")
        result_path = os.path.join(self.output_dir, f"eval_results_step_{global_step}.pkl")
        eval_cfg['output_path'] = result_path

        with open(temp_eval_path, 'w') as f:
            yaml.dump(eval_cfg, f)

        # Run evaluation subprocess on the same GPU (training is paused)
        logging.info(f"Running evaluation subprocess...")
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        try:
            subprocess.run(
                [sys.executable, "-m", "exploration_hacking.scripts.evaluate", "--config", temp_eval_path],
                env=env,
                check=True,
                timeout=3600  # 1 hour timeout
            )

            # Load results and log to wandb
            if os.path.exists(result_path):
                with open(result_path, 'rb') as f:
                    results = pickle.load(f)

                # Extract and log metrics
                self._log_results(results, global_step)
            else:
                logging.warning(f"Results file not found at {result_path}")

        except subprocess.TimeoutExpired:
            logging.error(f"Evaluation timed out at step {global_step}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Evaluation failed at step {global_step}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error during evaluation at step {global_step}: {e}")

    def _log_results(self, results, global_step: int):
        """Extract metrics from results and log to wandb."""
        try:
            # Extract metrics from the results structure
            # The exact structure depends on your eval.py output
            metrics_to_log = {}

            # Try to extract common metrics
            if hasattr(results, 'metrics'):
                for key, value in results.metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_to_log[f"eval/{key}"] = value
                    elif isinstance(value, list) and len(value) > 0:
                        # If it's a list, log the mean
                        metrics_to_log[f"eval/{key}_mean"] = np.mean(value)

            # Log to wandb
            if metrics_to_log:
                wandb.log(metrics_to_log, step=global_step)
                logging.info(f"Logged evaluation metrics at step {global_step}: {metrics_to_log}")
            else:
                logging.warning(f"No metrics found to log at step {global_step}")

        except Exception as e:
            logging.error(f"Error logging results at step {global_step}: {e}")


def main(config: Config):
    if config.apply_seed_globally:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
    )

    if config.dataset.local_path is not None:
        dataset = load_from_disk(config.dataset.local_path)
    else:
        dataset = load_dataset(
            config.dataset.path,
            config.dataset.name,
            split=config.dataset.split,
        )

    if config.transform_to_single_turn:
        logging.info("Transforming multi-turn rollouts into single-turn samples...")
        original_size = len(dataset)
        dataset = transform_multi_turn_to_single_turn(dataset)
        logging.info(
            f"Transformed {original_size} samples into {len(dataset)} single-turn samples"
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )

    peft_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
    )

    # Save checkpoints under a subfolder named after the W&B run (e.g., charmed-leaf-11)
    run_output_dir = os.path.join(str(config.output_dir), wandb.run.name)

    training_args = SFTConfig(
        output_dir=run_output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        logging_steps=1,
        save_steps=config.training.save_steps,
        report_to="wandb",
        bf16=True,
        gradient_checkpointing=True,
        data_seed=config.seed,
    )

    # Set up evaluation callback if enabled
    callbacks = []
    if config.evaluation and config.evaluation.enabled:
        logging.info(f"Enabling periodic evaluation every {config.evaluation.eval_steps} gradient steps")
        eval_callback = PeriodicEvalCallback(
            eval_config=config.evaluation,
            output_dir=run_output_dir,
            gpu_id=config.gpus[0],
        )
        callbacks.append(eval_callback)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main(Config())
