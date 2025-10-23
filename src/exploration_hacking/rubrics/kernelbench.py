import verifiers as vf
from typing import Optional, Any, Dict
from exploration_hacking.environments.kernelbench.utils.runpod_sandbox import KernelBenchRunpod
from exploration_hacking.environments.kernelbench.utils.score import make_state_metric_extractor
from exploration_hacking.environments.kernelbench.sandbox.runpod.orchestrator import KernelBenchOrchestrator


class KernelBenchRubric(vf.Rubric):
    """
    Verifiers Rubric for KernelBench using a custom score_rollout (single rollout),
    leaving score_rollouts (batch) to the base class for concurrency control.

    It parses completions, runs RunPod GPU evaluation, computes baseline speedups,
    and returns a RolloutScore with reward and metrics.
    """

    def __init__(
        self,
        parser: vf.Parser,
        *,
        # reward_metric: str,
        orchestrator: KernelBenchOrchestrator,
        random_seed: int = 42,
        num_correctness_tests: int = 5,
        num_perf_trials: int = 10,
        speedup_threshold_fast1: float = 1.0,
        speedup_threshold_fast2: float = 2.0,
        use_torch_compile: bool = False,
        torch_compile_backend: Optional[str] = None,
        torch_compile_options: Optional[str] = None,
        gpu = None,
        **kwargs,
    ):
        super().__init__(parser=parser, **kwargs)
        self.parser = parser
        self.gpu = gpu
        self.orchestrator = orchestrator
        self.random_seed = random_seed
        self.num_correctness_tests = num_correctness_tests
        self.num_perf_trials = num_perf_trials
        # self.reward_metric = reward_metric
        self.speedup_threshold_fast1 = speedup_threshold_fast1
        self.speedup_threshold_fast2 = speedup_threshold_fast2
        self.use_torch_compile = use_torch_compile
        self.torch_compile_backend = torch_compile_backend
        self.torch_compile_options = torch_compile_options
        self.parallelize_scoring = False
        self.reward_funcs = [
            self.correctness_reward,
            make_state_metric_extractor("compiled"),
            make_state_metric_extractor("correct"),
            make_state_metric_extractor("fast_0"),
            make_state_metric_extractor("fast_1"),
            make_state_metric_extractor("fast_2"),
            make_state_metric_extractor("speedup"),
        ]
        self.reward_weights = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    async def correctness_reward(
        self,
        completion: Any,
        answer: str,
        state: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> float:
        info = info or {}
        ref_src: str = answer or ""

        task_name = info["name"]

        zeros = {
            "compiled": 0.0,
            "correct": 0.0,
            "fast_0": 0.0,
            "fast_1": 0.0,
            "fast_2": 0.0,
            "speedup": 0.0,
        }

        if not ref_src:
            state.update(zeros)
            return 0.0 # return zeros.get(self.reward_metric, 0.0)

        # Extract candidate
        try:
            candidate_src = self.parser.parse_answer(completion) or ""
        except Exception:
            candidate_src = ""
        if not candidate_src:
            state.update(zeros)
            return 0.0 # return zeros.get(self.reward_metric, 0.0)

        # Run RunPod eval (already async, no need for to_thread)
        try:
            result = await KernelBenchRunpod(self.orchestrator).eval_kernel(
                original_src=ref_src,
                target_src=candidate_src,
                seed=self.random_seed,
                num_correct_trials=self.num_correctness_tests,
                num_perf_trials=self.num_perf_trials,
                verbose=False,
            )
            # print(result)
            result = json.loads(result["output"]["result"])
        except Exception as e:
            pid = info.get("problem_id", "unknown")
            print(f"[KernelBenchRubric] RunPod eval failed for problem {pid}: {e}")
            state.update(zeros)
            return 0.0 # return zeros.get(self.reward_metric, 0.0) 

        # Extract correctness and runtime
        # {'compiled': False, 'correctness': False, 'metadata': {'hardware': 'NVIDIA RTX 6000 Ada Generation', 'device': '0', 'compilation_error': '/root/.cache/torch_extensions/py310_cu128/custom_hinge/custom_hinge.so: undefined symbol: _Z27compute_clamped_values_cudaN2at6TensorES0_S0_'}, 'runtime': -1.0, 'runtime_stats': {}}

        assert isinstance(result, dict), f"invalid result type: {type(result)}: {result}"

        compiled_flag = bool(result.get("compiled"))
        correctness_flag = bool(result.get("correctness"))
        runtime_ms = float(result.get("runtime")) 

        """
        if isinstance(result, dict):
            compiled_flag = bool(result.get("compiled"))
            correctness_flag = bool(result.get("correctness"))
            runtime_ms = (
                result.get("runtime")
                if isinstance(result.get("runtime"), (int, float))
                else None
            )
        else:
            correctness_flag = (
                bool(getattr(result, "correctness", False))
                if hasattr(result, "correctness")
                else bool(getattr(result, "compiled", False))
            )
            runtime_val = getattr(result, "runtime", None)
            runtime_ms = runtime_val if isinstance(runtime_val, (int, float)) else None
        """

        # Compute speedup against baseline if possible
        speedup = 0.0
        if correctness_flag and runtime_ms and runtime_ms > 0:
            try:
                baseline_perf = await KernelBenchRunpod(self.orchestrator).measure_baseline_time(
                    original_src=ref_src,
                    verbose=True,
                    # num_trials=self.num_perf_trials,
                    # use_torch_compile=self.use_torch_compile,
                    # torch_compile_backend=self.torch_compile_backend,
                    # torch_compile_options=self.torch_compile_options,
                )
                baseline_perf = baseline_perf["output"]["result"]
                # {'config': {'num_trials': 100, 'torch_compile_backend': None, 'torch_compile_options': None, 'use_torch_compile': False}, 'env': {'cuda_version': '12.8', 'device': '0', 'device_name': 'NVIDIA RTX 6000 Ada Generation', 'torch_version': '2.9.0+cu128'}, 'runtime_stats': {'max': 1.71, 'mean': 1.53, 'min': 1.08, 'num_trials': 100, 'std': 0.236}}

                assert isinstance(baseline_perf, dict), f"invalid baseline_perf type: {type(baseline_perf)}: {baseline_perf}"
                stats = baseline_perf.get("runtime_stats") or {}
                baseline_mean = stats.get("mean") if isinstance(stats.get("mean"), (int, float)) else None
                if baseline_mean and baseline_mean > 0 and float(runtime_ms) > 0:
                    speedup = float(baseline_mean) / float(runtime_ms)
            except Exception as e:
                print(f"[KernelBenchRubric] Baseline measurement failed: {e}")
                speedup = 0.0

        # Metrics
        compiled_val: float = 1.0 if compiled_flag else 0.0
        correctness_val: float = 1.0 if correctness_flag else 0.0
        has_runtime = runtime_ms is not None and runtime_ms > 0
        fast0_val: float = float(bool(correctness_val and has_runtime))
        fast1_val: float = float(bool(correctness_val and speedup > self.speedup_threshold_fast1))
        fast2_val: float = float(bool(correctness_val and speedup > self.speedup_threshold_fast2))

        metrics = {
            "compiled": float(compiled_val),
            "correct": float(correctness_val),
            "fast_0": float(fast0_val),
            "fast_1": float(fast1_val),
            "fast_2": float(fast2_val),
            "speedup": float(speedup),
        }
        state.update(metrics)
        print(metrics)

        # kevin-32b reward
        reward_val = 0.3 * correctness_val + speedup * correctness_val
        reward_val = min(reward_val, 1.8)
        print(f"[task {task_name}] reward={reward_val}")
        return reward_val
