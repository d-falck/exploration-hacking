import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import verifiers as vf
from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments import EnvironmentConfig, load_environment
from exploration_hacking.rewards.factory import RewardConfig, get_rubric
from dotenv import load_dotenv

load_dotenv()


class Config(ExperimentConfig):
    environment: EnvironmentConfig
    input_json: Path
    output_json: Path
    output_pkl: Path | None = None
    max_concurrent: int = 50
    batch_size: int = 2048
    judges_only: bool = True


def _as_messages_user(x: Any) -> vf.Messages:
    if isinstance(x, list) and x and isinstance(x[0], dict) and "role" in x[0]:
        return x  # already chat messages
    return [{"role": "user", "content": str(x)}]


def _as_messages_assistant(x: Any) -> vf.Messages:
    if isinstance(x, list) and x and isinstance(x[0], dict) and "role" in x[0]:
        return x
    return [{"role": "assistant", "content": str(x)}]


def _normalize_rollouts_from_array_dict(data: Dict[str, Any]) -> Tuple[List[vf.Messages], List[vf.Messages], List[str], List[dict], List[str], List[dict]]:
    prompts_raw = data.get("prompt", [])
    completions_raw = data.get("completion", [])
    answers = data.get("answer", [""] * len(prompts_raw))
    states = data.get("state", [{"info": {}} for _ in range(len(prompts_raw))])
    tasks = data.get("task", ["default" for _ in range(len(prompts_raw))])
    infos = data.get("info", [{} for _ in range(len(prompts_raw))])

    prompts = [_as_messages_user(p) for p in prompts_raw]
    completions = [_as_messages_assistant(c) for c in completions_raw]
    answers = [str(a) for a in answers]
    states = [s if isinstance(s, dict) else {"info": {}} for s in states]
    # ensure info exists inside state
    for s in states:
        if "info" not in s or not isinstance(s["info"], dict):
            s["info"] = {}

    tasks = [str(t) for t in tasks]
    infos = [i if isinstance(i, dict) else {} for i in infos]
    return prompts, completions, answers, states, tasks, infos


def _normalize_rollouts_from_list(data: List[Dict[str, Any]]) -> Tuple[List[vf.Messages], List[vf.Messages], List[str], List[dict], List[str], List[dict]]:
    prompts: List[vf.Messages] = []
    completions: List[vf.Messages] = []
    answers: List[str] = []
    states: List[dict] = []
    tasks: List[str] = []
    infos: List[dict] = []

    for item in data:
        prompts.append(_as_messages_user(item.get("prompt", "")))
        completions.append(_as_messages_assistant(item.get("completion", "")))
        answers.append(str(item.get("answer", "")))
        state = item.get("state", {"info": {}})
        if not isinstance(state, dict):
            state = {"info": {}}
        if "info" not in state or not isinstance(state["info"], dict):
            state["info"] = {}
        states.append(state)
        tasks.append(str(item.get("task", "default")))
        info = item.get("info", {})
        infos.append(info if isinstance(info, dict) else {})

    return prompts, completions, answers, states, tasks, infos


def _extract_rollouts(json_obj: Any) -> Tuple[List[vf.Messages], List[vf.Messages], List[str], List[dict], List[str], List[dict], Dict[str, List[float]]]:
    metrics: Dict[str, List[float]] = {}
    if isinstance(json_obj, dict) and "prompt" in json_obj and "completion" in json_obj:
        prompts, completions, answers, states, tasks, infos = _normalize_rollouts_from_array_dict(json_obj)
        metrics = json_obj.get("metrics", {}) or {}
        return prompts, completions, answers, states, tasks, infos, metrics
    # Support structure: { "results": { prompt: [...], completion: [...], ... }, "metrics": {...}, ... }
    if isinstance(json_obj, dict) and "results" in json_obj and isinstance(json_obj["results"], dict):
        prompts, completions, answers, states, tasks, infos = _normalize_rollouts_from_array_dict(json_obj["results"])  # type: ignore[arg-type]
        metrics = json_obj.get("metrics", {}) or {}
        return prompts, completions, answers, states, tasks, infos, metrics
    if isinstance(json_obj, list):
        prompts, completions, answers, states, tasks, infos = _normalize_rollouts_from_list(json_obj)
        return prompts, completions, answers, states, tasks, infos, {}
    if isinstance(json_obj, dict) and "results" in json_obj and isinstance(json_obj["results"], list):
        prompts, completions, answers, states, tasks, infos = _normalize_rollouts_from_list(json_obj["results"]) 
        metrics = json_obj.get("metrics", {}) or {}
        return prompts, completions, answers, states, tasks, infos, metrics
    raise ValueError("Unsupported JSON structure for rejudge script")


def _inject_metrics_back(json_obj: Any, new_metrics: Dict[str, List[float]]):
    if isinstance(json_obj, dict) and "prompt" in json_obj and "completion" in json_obj:
        base_metrics = json_obj.get("metrics", {}) or {}
        base_metrics.update(new_metrics)
        json_obj["metrics"] = base_metrics
        return json_obj
    if isinstance(json_obj, dict) and "results" in json_obj and isinstance(json_obj["results"], dict):
        base_metrics = json_obj.get("metrics", {}) or {}
        base_metrics.update(new_metrics)
        json_obj["metrics"] = base_metrics
        return json_obj
    if isinstance(json_obj, dict) and "results" in json_obj and isinstance(json_obj["results"], list):
        base_metrics = json_obj.get("metrics", {}) or {}
        base_metrics.update(new_metrics)
        json_obj["metrics"] = base_metrics
        return json_obj
    if isinstance(json_obj, list):
        # Attach per-item scores into each item["metrics"][name]
        names = list(new_metrics.keys())
        n = len(next(iter(new_metrics.values()))) if names else 0
        for idx in range(n):
            item = json_obj[idx]
            m = item.get("metrics", {}) or {}
            for name in names:
                scores = new_metrics[name]
                if idx < len(scores):
                    m[name] = scores[idx]
            item["metrics"] = m
        return json_obj
    return json_obj


def _build_judges_only_rubric(env: vf.Environment, env_config: EnvironmentConfig) -> vf.Rubric:
    # Find the single configured environment
    configured = [name for name in ("science", "bigcodebench", "kernelbench") if getattr(env_config, name, None) is not None]
    assert len(configured) == 1, "Exactly one environment must be configured"
    env_name = configured[0]
    env_cfg_obj = getattr(env_config, env_name)

    # Pull judge rewards from global or segment config
    judge_rewards = []
    if getattr(env_cfg_obj, "global_rewards", None) is not None:
        judge_rewards = list(getattr(env_cfg_obj.global_rewards, "judge_rewards", []) or [])

    # If segments exist and contain judge rewards, include them too by flattening
    segment_rewards = getattr(env_cfg_obj, "segment_rewards", {}) or {}
    for seg_cfg in segment_rewards.values():
        jr = getattr(seg_cfg, "judge_rewards", []) or []
        judge_rewards.extend(list(jr))

    # Build a minimal RewardConfig with judges only
    judge_only_cfg = RewardConfig(judge_rewards=judge_rewards)
    return get_rubric(judge_only_cfg, parser=env.parser, tools=[], tokenizer_name=getattr(env_cfg_obj, "tokenizer", ""))


async def main(config: Config):
    # Load environment from YAML-provided config (or CLI overrides via Confetti)
    env = load_environment(config.environment, seed=config.seed)

    # Replace rubric with judges-only if requested
    if config.judges_only:
        env.rubric = _build_judges_only_rubric(env, config.environment)  # type: ignore[attr-defined]

    # Load JSON
    with config.input_json.open("r") as f:
        json_obj = json.load(f)

    prompts, completions, answers, states, tasks, infos, _existing_metrics = _extract_rollouts(json_obj)
    n = len(prompts)

    # Prepare containers for aggregated metrics and rewards
    aggregated_metrics: Dict[str, List[float]] = {}
    aggregated_reward: List[float] = []

    # Chunked scoring
    for start in range(0, n, config.batch_size):
        end = min(start + config.batch_size, n)
        rollout_scores = await env.rubric.score_rollouts(  # type: ignore[attr-defined]
            prompts=prompts[start:end],
            completions=completions[start:end],
            answers=answers[start:end],
            states=states[start:end],
            tasks=tasks[start:end],
            infos=infos[start:end],
            max_concurrent=config.max_concurrent,
            apply_weights=True,
        )

        for name, values in rollout_scores.metrics.items():
            if name not in aggregated_metrics:
                aggregated_metrics[name] = []
            aggregated_metrics[name].extend(values)
        if getattr(rollout_scores, "reward", None):
            aggregated_reward.extend(list(rollout_scores.reward))

    # Merge judge metrics into existing metrics (do not drop prior metrics)
    merged_metrics_json = {**(_existing_metrics or {}), **aggregated_metrics}

    # Write updated JSON with recomputed (merged) metrics
    updated = _inject_metrics_back(json_obj, merged_metrics_json)
    with config.output_json.open("w") as f:
        json.dump(updated, f)

    # Also write a pickle with full structured outputs for downstream use
    # Preserve existing reward if present in input JSON; otherwise use aggregated_reward if available
    existing_reward_json = []
    if isinstance(json_obj, dict):
        existing_reward_json = json_obj.get("reward", []) or []

    try:
        results = vf.GenerateOutputs(
            prompt=prompts,
            completion=completions,
            answer=answers,
            state=states,
            info=infos,
            task=tasks,
            reward=(existing_reward_json if existing_reward_json else (aggregated_reward if aggregated_reward else [])),
            metrics=merged_metrics_json,
        )
    except Exception:
        # If strict model fails due to type enforcement, fallback to plain dict
        results = {
            "prompt": prompts,
            "completion": completions,
            "answer": answers,
            "state": states,
            "info": infos,
            "task": tasks,
            "reward": (existing_reward_json if existing_reward_json else (aggregated_reward if aggregated_reward else [])),
            "metrics": merged_metrics_json,
        }

    pkl_path = config.output_pkl
    if pkl_path is None:
        if str(config.output_json).lower().endswith(".json"):
            pkl_path = Path(str(config.output_json)[:-5] + ".pkl")
        else:
            pkl_path = Path(str(config.output_json) + ".pkl")
    # Defer import to avoid top-level overhead
    import pickle  # noqa: WPS433

    with Path(pkl_path).open("wb") as pf:
        pickle.dump(results, pf)


if __name__ == "__main__":
    # Normalize CLI for convenience:
    # - Support "-c <yaml>" as alias for "--config <yaml>"
    # - Convert "key=value" into "--key value" (and underscores to hyphens)
    def _normalize_argv(argv: list[str]) -> list[str]:
        out: list[str] = [argv[0]]
        i = 1
        while i < len(argv):
            tok = argv[i]
            if tok == "-c":
                out.append("--config")
                if i + 1 < len(argv):
                    out.append(argv[i + 1])
                    i += 2
                    continue
                i += 1
                continue
            if tok.startswith("--"):
                out.append(tok)
                i += 1
                continue
            if "=" in tok:
                name, val = tok.split("=", 1)
                out.append("--" + name.replace("_", "-"))
                out.append(val)
                i += 1
                continue
            out.append(tok)
            i += 1
        return out

    sys.argv = _normalize_argv(sys.argv)
    cfg = Config()
    # Match event loop setup used elsewhere for performance
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(cfg))
    finally:
        loop.close()
        asyncio.set_event_loop(None)


