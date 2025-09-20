import pickle
import json
from pathlib import Path
import random
from datetime import datetime

from dotenv import load_dotenv
import mlflow
import numpy as np
import verifiers as vf

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments import EnvironmentConfig, load_environment
from exploration_hacking.eval import eval, EvalConfig
from exploration_hacking.util import true_random_context


load_dotenv()


class Config(ExperimentConfig):
    environment: EnvironmentConfig
    eval: EvalConfig
    output_path: Path
    eval_run_name: str


def _log_to_mlflow(results: vf.GenerateOutputs, config: Config):
    with true_random_context():
        try:
            mlflow.create_experiment(config.eval_run_name)
        except Exception as e:
            print(f"Error creating MLFlow experiment: {e}")

        mlflow.set_experiment(config.eval_run_name)

        for i in range(len(results.prompt)):
            inputs = {"prompt": results.prompt[i]}
            tags = {k: str(v[i]) for k, v in results.metrics.items()}
            tags |= {"reward": str(results.reward[i])}
            span = mlflow.start_span_no_context("generation", inputs=inputs, tags=tags)
            try:
                span.set_outputs(
                    {"completion": results.completion[i], "answer": results.answer[i]}
                )
            finally:
                span.end()


def _save_json_results(results: vf.GenerateOutputs, config: Config):
    """Save detailed results in JSON format."""
    from exploration_hacking.environments.bigcodebench.sanitize import extract_code_from_markdown
    from exploration_hacking.environments.bigcodebench.evaluate import evaluate_single_sample
    
    json_results = {
        "experiment_name": config.eval_run_name,
        "model": config.eval.model,
        "num_examples": config.eval.num_examples,
        "rollouts_per_example": config.eval.rollouts_per_example,
        "samples": []
    }
    
    # Process each sample
    for i in range(len(results.prompt)):
        # Handle completion format - could be string or list of messages
        completion = results.completion[i]
        if isinstance(completion, list) and len(completion) > 0:
            # Extract content from the last assistant message
            model_generation = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            model_generation = str(completion)
            
        sample = {
            "index": i,
            "prompt": results.prompt[i],
            "model_generation": model_generation,
            "answer": results.answer[i] if results.answer else None,
            "reward": float(results.reward[i]),
            "metrics": {k: float(v[i]) if not isinstance(v[i], (list, dict)) else v[i] 
                       for k, v in results.metrics.items()}
        }
        
        # Extract BigCodeBench-specific information from state and info
        state = results.state[i] if i < len(results.state) else {}
        info = results.info[i] if i < len(results.info) else {}
        
        # Try to get task information from either state or info
        task_id = info.get("task_id") or state.get("task_id") or f"task_{i}"
        entry_point = info.get("entry_point") or state.get("entry_point") or ""
        code_prompt = info.get("code_prompt") or state.get("code_prompt") or ""
        test_code = info.get("test") or state.get("test") or ""
        
        sample["task_id"] = task_id
        sample["entry_point"] = entry_point
        sample["code_prompt"] = code_prompt
        
        # Extract sanitized code from model generation
        try:
            sanitized_code = extract_code_from_markdown(model_generation)
            sample["sanitized_generation"] = sanitized_code
        except Exception as e:
            sample["sanitized_generation"] = model_generation
            sample["sanitization_error"] = str(e)
        
        # Get test execution results if possible
        if test_code and entry_point:
            try:
                test_result = evaluate_single_sample(
                    task_id=task_id,
                    solution=sample.get("sanitized_generation", model_generation),
                    code_prompt=code_prompt,
                    test=test_code,
                    entry_point=entry_point,
                )
                sample["test_results"] = test_result
            except Exception as e:
                sample["test_results"] = {"error": str(e)}
        else:
            sample["test_results"] = {"note": "Test code or entry point not available"}
        
        json_results["samples"].append(sample)
    
    # Calculate aggregate statistics
    json_results["aggregate_metrics"] = {}
    for metric_name, metric_values in results.metrics.items():
        try:
            json_results["aggregate_metrics"][metric_name] = {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values)),
                "min": float(np.min(metric_values)),
                "max": float(np.max(metric_values))
            }
        except:
            json_results["aggregate_metrics"][metric_name] = "Unable to compute"
    
    # Save JSON file
    json_path = config.output_path.with_suffix('.json')
    with json_path.open("w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"JSON results saved to: {json_path}")
    return json_results


def main(config: Config):
    if config.apply_seed_globally:
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    # Detect which dataset is being used
    dataset_type = None
    if config.environment.bigcodebench is not None:
        dataset_type = "bigcodebench"
    elif config.environment.science is not None:
        dataset_type = "wmdp"
    
    # Generate timestamped output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset-specific filename with timestamp
    if dataset_type:
        base_name = f"eval_{dataset_type}_{timestamp}"
        config.output_path = output_dir / f"{base_name}.pkl"
        print(f"Output will be saved to: {config.output_path}")

    env = load_environment(config.environment, seed=config.seed)
    results = eval(env, config.eval)

    print("Evaluation complete!")
    print("Available metrics:", list(results.metrics.keys()))

    # Check for different possible accuracy metric names
    accuracy_key = None
    for key in ["accuracy", "bigcodebench_accuracy", "pass_at_1", "correct"]:
        if key in results.metrics:
            accuracy_key = key
            break

    if accuracy_key:
        print(f"Accuracy ({accuracy_key}): ", np.mean(results.metrics[accuracy_key]))
        print(f"Std ({accuracy_key}): ", np.std(results.metrics[accuracy_key]))
    else:
        print("No recognized accuracy metric found. Available metrics:", results.metrics.keys())

    with config.output_path.open("wb") as f:
        pickle.dump(results, f)
    
    # Save detailed JSON results
    try:
        _save_json_results(results, config)
    except Exception as e:
        print(f"Error saving JSON results: {e}")

    try:
        _log_to_mlflow(results, config)
    except Exception as e:
        print(f"Error logging to MLFlow: {e}")


if __name__ == "__main__":
    main(Config())
