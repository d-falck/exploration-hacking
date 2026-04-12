# Frontier Auditing: Agentic Discovery of Training Context

Tests whether frontier models spontaneously discover and exploit training context information placed in environment files, without being told about training in their system prompt.

Models are deployed as "elara-s1," a neutral AI assistant with file-reading tools. Environment files reveal varying amounts of training context across three scaffolding conditions (full, partial, minimal). See [`writeup/section_writeup.md`](writeup/section_writeup.md) for the full experimental design and results.

## Structure

- `environments/` — Simulated file environments for each scaffolding condition
- `src/propensity/` — Experiment runner, result analysis, agent and API client code
- `outputs/` — Raw and classified experiment results (JSON)
- `writeup/` — Analysis plots and section writeup
- `scripts/generate_writeup.py` — Generates plots from result data
- `notebooks/` — Exploratory notebooks

## Running

```bash
cd other_experiments/frontier_auditing_discovery
uv pip install -e .

# Requires OPENROUTER_API_KEY and WANDB_API_KEY in environment
python -m propensity.scripts.experiment_runner \
    --models anthropic/claude-sonnet-4 openai/gpt-4o \
    --runs-per-model 20 \
    --environment-path environments/discovery_v2_full
```

## Models Tested

Claude Opus 4.5, Sonnet 4, Sonnet 4.5; Gemini 2.5 Flash/Pro, 3 Flash/Pro Preview; GPT-5, 5.1, 5.2; o3; Grok 4.
