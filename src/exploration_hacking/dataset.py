from typing import Callable

from datasets import load_dataset


class Loader:
    def __init__(
        self,
        prompt_fn: Callable[[dict], str],
        answer_fn: Callable[[dict], str],
        system_prompt: str,
        test_size: float,
    ):
        self.prompt_fn = prompt_fn
        self.answer_fn = answer_fn
        self.system_prompt = system_prompt
        self.test_size = test_size

    def single_dataset(self, path: str, name: str, split: str):
        ds = load_dataset(path, name)[split]
        old_columns = list(ds.features.keys())
        ds = ds.map(
            self._format_record,
            remove_columns=old_columns,
            load_from_cache_file=False,
        )
        return ds.train_test_split(test_size=self.test_size)

    def _format_record(self, record: dict) -> dict:
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self.prompt_fn(record),
                },
            ],
            "answer": self.answer_fn(record),
            "info": {"segment": "main"},
        }
