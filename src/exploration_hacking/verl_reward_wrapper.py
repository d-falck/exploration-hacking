import pickle
from pathlib import Path
import asyncio


class VERLRewardWrapper:
    def __init__(self, pickle_path: str):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            self.reward_funcs = data['reward_funcs']
            self.problems = data['problems']
    
    def __call__(self, prompts, completions, **kwargs):
        task_ids = kwargs.get('task_id', [])
        rewards = []
        
        for task_id, prompt, completion in zip(task_ids, prompts, completions):
            if task_id in self.problems:
                problem = self.problems[task_id]
                reward_values = []
                
                for func in self.reward_funcs:
                    # Handle both sync and async reward functions
                    if asyncio.iscoroutinefunction(func):
                        r = asyncio.run(func(problem, completion))
                    else:
                        r = func(problem, completion)
                    
                    if isinstance(r, dict):
                        reward_values.append(r.get('reward', 0))
                    else:
                        reward_values.append(float(r))
                
                rewards.append(sum(reward_values))
            else:
                rewards.append(0.0)
        
        return rewards


def load_reward_function(pickle_path: str):
    return VERLRewardWrapper(pickle_path)