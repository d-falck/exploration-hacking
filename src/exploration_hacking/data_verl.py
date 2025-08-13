from pathlib import Path
import pandas as pd
from datasets import Dataset


def to_parquet(dataset: Dataset, output_path: str, problems=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = dataset.to_pandas()
    
    if problems and "task_id" in df.columns:
        metadata = []
        for task_id in df["task_id"]:
            if task_id in problems:
                problem = problems[task_id]
                metadata.append({
                    "task_id": task_id,
                    "difficulty": getattr(problem, "difficulty", None),
                })
            else:
                metadata.append({"task_id": task_id})
        df["metadata"] = metadata
    
    df.to_parquet(output_path, engine="pyarrow")
    return str(output_path)