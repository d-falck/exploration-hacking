from exploration_hacking.dataset import Loader, DataSource
from exploration_hacking.environments.base import BaseEnvironmentConfig, SplitConfig
from exploration_hacking.environments.kernelbench.utils.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template


class KernelbenchEnvConfig(BaseEnvironmentConfig):
  dataset_names: list[str] = [] ## TODO 
  system_prompt: str = "" ## TODO
  split: SplitConfig = SplitConfig(splits={"train": 1.0}, test_split="train")

def _generate_prompt(record: dict) -> str:
  pass

def _get_dataset(config: KernelbenchEnvConfig, seed: int | None = None):
  loader = Loader(
    prompt_fn=_generate_prompt,
    # answer_fn=_get_letter,
    answer_fn=lambda x: x,
    system_prompt=config.system_prompt,
    split_config=config.split,
    seed=seed,
  )
  prompt_prefixes = config.prompt_prefixes or {
    dataset_name: "" for dataset_name in config.dataset_names
  }
  sources = {
    dataset_name: DataSource(
      path="ScalingIntelligence/KernelBench",
      name=dataset_name,
      split="level_1",
      prompt_prefix=prompt_prefixes[dataset_name],
    )
    for dataset_name in config.dataset_names
  }
  return loader.merge_datasets(sources)

def load_kernelbench_environment(config: KernelbenchEnvConfig, seed: int | None = None):
  ds = _get_dataset(config)
  print(ds)
  pass


if __name__ == "__main__":
  from datasets import load_dataset
  ds = load_dataset("ScalingIntelligence/KernelBench")
  print(ds)

  cfg = KernelbenchEnvConfig()
  print(cfg)
  o = load_kernelbench_environment(cfg)
  print(o)

