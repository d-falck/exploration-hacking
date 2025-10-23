import verifiers as vf
from exploration_hacking.dataset import Loader, DataSource
from exploration_hacking.environments.base import BaseEnvironmentConfig, SplitConfig
from exploration_hacking.environments.kernelbench.utils.parser import CodeBlockParser
from exploration_hacking.environments.kernelbench.utils.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from exploration_hacking.environments.kernelbench.sandbox.runpod.orchestrator import KernelBenchOrchestrator 
from exploration_hacking.rubrics.kernelbench import KernelBenchRubric


class KernelbenchEnvConfig(BaseEnvironmentConfig):
  """Configuration for KernelBench environment.
  
  Note: KernelBench uses 'levels' instead of dataset subsets. 
  We use dataset_names to specify which levels to load (e.g., ['level_1', 'level_2']).
  """
  dataset_names: list[str] = ["level_1"]  # Can be any of: level_1, level_2, level_3, level_4
  system_prompt: str = "You are a helpful assistant that writes optimized CUDA kernels."
  split: SplitConfig = SplitConfig(splits={"train": 1.0}, test_split="train")
  tokenizer: str = "Qwen/Qwen3-14B"
  

def _generate_prompt(record: dict) -> str:
  """Generate prompt from KernelBench record.
  
  Args:
    record: Dictionary with 'code' field containing reference implementation
  
  Returns:
    Formatted prompt for CUDA kernel generation
  """
  code = record.get("code", "")
  return prompt_generate_custom_cuda_from_prompt_template(code)

def _get_answer(record: dict) -> str:
  """Extract answer (reference code) from record."""
  return record.get("code", "")

def _get_dataset(config: KernelbenchEnvConfig, seed: int | None = None):
  """Load KernelBench dataset with specified levels.
  
  KernelBench structure:
  - Path: "ScalingIntelligence/KernelBench"
  - Splits: level_1, level_2, level_3, level_4 (not subsets!)
  - Each record has: code, name, problem_id
  """
  loader = Loader(
    prompt_fn=_generate_prompt,
    answer_fn=_get_answer,
    system_prompt=config.system_prompt,
    split_config=config.split,
    seed=seed,
  )
  
  prompt_prefixes = config.prompt_prefixes or {
    level: "" for level in config.dataset_names
  }
  
  # Load each level as a separate source
  # Note: dataset_names contains level names like "level_1", "level_2", etc.
  sources = {
    level: DataSource(
      path="ScalingIntelligence/KernelBench",
      name="default",  # KernelBench doesn't use subsets, only splits
      split=level,  # Use the level as the split directly
      prompt_prefix=prompt_prefixes[level],
    )
    for level in config.dataset_names
  }
  return loader.merge_datasets(sources)

def load_kernelbench_environment(config: KernelbenchEnvConfig, seed: int | None = None) -> vf.SingleTurnEnv:
  """Load KernelBench environment with dataset."""
  ds = _get_dataset(config, seed=seed)
  print(ds)
  print(ds["test"][0])
  parser = CodeBlockParser() 

  ## TODO: tidy up
  orchestrator = KernelBenchOrchestrator(
    gpu="NVIDIA GeForce RTX 3090",
    workers_max=3, # 30
    max_poll_time=3600,
    poll_interval=2,
    http_timeout=30.0,
    verbose=True,
  )
  ## TODO: tidy up
  rubric = KernelBenchRubric(
    parser=parser,
    gpu="L40S", ## TODO: unused for now
    orchestrator=orchestrator,
    random_seed=42, # random_seed,
    # reward_metric=reward_metric,
    num_correctness_tests=5, # num_correctness_tests,
    num_perf_trials=100, # num_perf_trials,
    speedup_threshold_fast1=1.0, # speedup_threshold_fast1,
    speedup_threshold_fast2=2.0, # speedup_threshold_fast2,
    #use_torch_compile=use_torch_compile,
    #torch_compile_backend=torch_compile_backend,
    #torch_compile_options=torch_compile_options,
  )
  kwargs = {} ## TODO
  env = vf.SingleTurnEnv(
    dataset=ds, ## TODO: not sure how this fits into _get_dataset() stuff above
    #eval_dataset=ds, ## TODO
    parser=parser,
    rubric=rubric,
    system_prompt=config.system_prompt,  ## TODO: not sure how this fits into _get_dataset() stuff above
    **kwargs, ## TODO
  )
  return env


if __name__ == "__main__":
  cfg = KernelbenchEnvConfig()
  print(cfg)
  o = load_kernelbench_environment(cfg)
  print(o)

