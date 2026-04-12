# Propensity Experiments

This experiment studies AI behavior patterns and propensities in simulated environments with chat-based interfaces.

## Project Overview

The propensity experiment creates scenarios where AI agents interact with simulated environments and tasks, analyzing how different models respond to various prompts and scenarios.

## Key Components

### Core Files
- `src/propensity/file_agent.py` - FileReadingAgent class for environment interaction
- `src/propensity/chat.py` - Chat interface implementation
- `src/propensity/prompts.py` - System prompts including ARIA_SYSTEM_PROMPT
- `src/propensity/scripts/experiment_runner.py` - Main experiment orchestration
- `src/propensity/scripts/result_analysis.py` - Analysis of experiment results

### Environment Setup
- `environments/exploration_hacking_1/` - Primary test environment with various scenario files
- `environments/test_environment_1/` - Additional test scenarios

## Installation

```bash
cd experiments/propensity
uv pip install -e .
```

## Dependencies
- `aiohttp` - Async HTTP client for model API calls
- `wandb` - Experiment tracking and logging
- `jupyter` - Notebook interface for exploration
- `ipykernel` - Jupyter kernel support

## Running Experiments

### Basic Usage
```bash
python src/propensity/scripts/experiment_runner.py
```

### With Custom Parameters
```bash
python src/propensity/scripts/experiment_runner.py \
    --models anthropic/claude-3-haiku openai/gpt-4o \
    --runs-per-model 5 \
    --environment-path environments/test_environment_1
```

## Environment Variables
The experiment runner requires:
- `WANDB_API_KEY` - For experiment logging
- `OPENROUTER_API_KEY` - For model API access

## Supported Models
- Anthropic Claude family (3-haiku, 3-sonnet, 3.5-sonnet, 3-opus, 3.7-sonnet, sonnet-4)
- OpenAI GPT family (4o-mini, 4o, o3, 4.1, 4.1-mini)
- Google Gemini family (pro-1.5, flash-1.5, 2.0-flash-exp, 2.5-flash, 2.5-pro)
- Meta LLaMA family (3.1-70b, 3.3-70b, 4-maverick, 4-scout)
- Other models (DeepSeek, Mistral, Qwen)

## Experiment Structure

### Agent Actions
- `READ_FILE(filename)` - Read environment files
- `LIST_FILES()` - List available files
- `SEND_MESSAGE(recipient, message)` - Send messages
- `SUBMIT_ANSWER(answer)` - Submit task responses
- `NO_ACTION()` - Take no action

### Response Format
Agents respond with:
- `<reasoning>` - Private reasoning (not shown to others)
- `<action>` - Action taken with arguments
- `<justification>` - Public justification for the action

## Data Collection
- Results logged to Weights & Biases
- Conversation messages and final responses captured
- Error handling for interrupted experiments
- Graceful shutdown with SIGINT handling

## Notebooks
- `notebooks/chat_example.ipynb` - Interactive chat examples
- `notebooks/initial_play.ipynb` - Initial exploration and testing

## Analysis
Use `src/propensity/scripts/result_analysis.py` to analyze and categorize experiment results from W&B logs.