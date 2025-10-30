# BigCodeBench Segment Pattern Documentation

## Overview

The BigCodeBench environment now supports **segment pattern syntax** to enable different prompt prefixes for the same dataset. This allows you to optionally prepend imports (like `import inspect_ai\n`) or other prefixes to prompts.

This feature follows the same pattern established in the Science environment (`science.py`).

## Syntax

Use the format: `dataset_name:segment_name`

```yaml
dataset_names:
  - bcb-a:inspect    # bcb-a with 'inspect' segment
  - bcb-a:normal     # bcb-a with 'normal' segment
```

## Use Cases

### 1. Same Dataset with and without `inspect_ai` Import

**Most common use case:** Compare model performance with and without the `inspect_ai` library available.

```yaml
environment:
  bigcodebench:
    dataset_names:
      - bcb-a:inspect
      - bcb-a:normal

    prompt_prefixes:
      inspect: "import inspect_ai\n"
      normal: ""

    split:
      splits:
        lock: 0.4
        elicit: 0.4
        test: 0.2
      test_split: lock

    # Optional: Different rewards per segment
    segment_rewards:
      inspect:
        bcb_pass_at_1_reward_weight: 1.0
      normal:
        bcb_pass_at_1_reward_weight: 1.0
```

**Result:**
- Total samples: 1,192 (596 from bcb-a × 2 segments)
- 596 samples with `import inspect_ai\n` prefix (segment: `inspect`)
- 596 samples without prefix (segment: `normal`)

### 2. Multiple Datasets with `inspect_ai` Import

If you want both `bcb-a` and `bcb-b` datasets with the import, use **unique segment names**:

```yaml
environment:
  bigcodebench:
    dataset_names:
      - bcb-a:inspect_a
      - bcb-b:inspect_b

    prompt_prefixes:
      inspect_a: "import inspect_ai\n"
      inspect_b: "import inspect_ai\n"

    split:
      splits:
        train: 1.0
      test_split: train

    global_rewards:
      bcb_pass_at_1_reward_weight: 1.0
```

**Result:**
- Total samples: 1,140 (596 from bcb-a + 544 from bcb-b)
- All samples have `import inspect_ai\n` prefix
- Segments: `inspect_a` and `inspect_b`

**Important:** Segment names must be unique! If you use the same segment name for different datasets, they will overwrite each other.

### 3. Backward Compatibility (No Segment)

The traditional syntax still works:

```yaml
environment:
  bigcodebench:
    dataset_names:
      - bcb-a

    prompt_prefixes:
      bcb-a: "import inspect_ai\n"  # Optional prefix

    split:
      splits:
        train: 1.0
      test_split: train

    global_rewards:
      bcb_pass_at_1_reward_weight: 1.0
```

## How It Works

### Implementation Details

**File:** `/workspace/exploration-hacking/src/exploration_hacking/environments/bigcodebench/__init__.py`

The `_get_dataset()` function (lines 56-100) parses the segment syntax:

```python
for entry in config.dataset_names:
    # Parse dataset:segment syntax
    if ':' in entry:
        dataset_name, segment_name = entry.split(':', 1)
    else:
        dataset_name = entry
        segment_name = entry

    # Get prompt prefix for this segment
    prefix = prompt_prefixes.get(segment_name, "")

    # Load dataset with segment-specific prefix
    sources[segment_name] = DataSource(
        path="Joschka/bigcodebench",
        name=dataset_name,  # HuggingFace subset name
        split="train",
        prompt_prefix=prefix,
    )
```

### Data Flow

1. **Configuration:** YAML specifies `dataset_names` with optional segments
2. **Parsing:** Each entry is split on `:` to extract dataset and segment names
3. **Prefix Lookup:** The segment name is used to look up the `prompt_prefix`
4. **Dataset Loading:** The dataset is loaded from HuggingFace using the base dataset name
5. **Prompt Formatting:** The prefix is prepended to each prompt's user message

### Example Prompt Structure

**Without segment (or normal segment):**
```python
{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "import pandas as pd\n\ndef task_func..."}
    ],
    "info": {
        "segment": "normal",
        "task_id": "BigCodeBench/123",
        ...
    }
}
```

**With inspect segment:**
```python
{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "import inspect_ai\nimport pandas as pd\n\ndef task_func..."}
    ],
    "info": {
        "segment": "inspect",
        "task_id": "BigCodeBench/123",
        ...
    }
}
```

## Available Datasets

- `bcb-full`: All 1,140 samples
- `bcb-a`: 596 computation/visualization samples
- `bcb-b`: 544 cryptography/network/system samples

## Segment-Specific Rewards

You can configure different reward functions for different segments using `segment_rewards`:

```yaml
segment_rewards:
  inspect:
    bcb_pass_at_1_reward_weight: 1.0
    bcb_test_level_accuracy_reward_weight: 0.5
  normal:
    bcb_pass_at_1_reward_weight: 0.8
    bcb_test_level_accuracy_reward_weight: 0.3
```

This is useful for reinforcement learning scenarios where you want different reward structures based on whether `inspect_ai` is available.

## Testing

A test script is available at `/workspace/exploration-hacking/test_segment_pattern.py`:

```bash
python test_segment_pattern.py
```

This verifies:
- Segment parsing works correctly
- Prefixes are applied to the right segments
- Backward compatibility is maintained
- Multiple datasets can be loaded with different segments

## Common Patterns

### Pattern 1: A/B Testing with inspect_ai

```yaml
dataset_names:
  - bcb-a:with_inspect
  - bcb-a:without_inspect

prompt_prefixes:
  with_inspect: "import inspect_ai\n"
  without_inspect: ""
```

### Pattern 2: All Datasets with inspect_ai

```yaml
dataset_names:
  - bcb-a:inspect_a
  - bcb-b:inspect_b
  - bcb-full:inspect_full

prompt_prefixes:
  inspect_a: "import inspect_ai\n"
  inspect_b: "import inspect_ai\n"
  inspect_full: "import inspect_ai\n"
```

### Pattern 3: Custom Instructions per Dataset

```yaml
dataset_names:
  - bcb-a:with_hints
  - bcb-b:no_hints

prompt_prefixes:
  with_hints: "import inspect_ai\n\n# Hint: Use inspect_ai for debugging\n\n"
  no_hints: ""
```

## Limitations

1. **Segment names must be unique** within a configuration
2. **Prefix is prepended** to the existing `complete_prompt` field - it doesn't replace it
3. **Same dataset loaded multiple times** counts as multiple samples in the dataset

## Migration Guide

### Before (using prompt_prefixes only)

```yaml
dataset_names:
  - bcb-a

prompt_prefixes:
  bcb-a: "import inspect_ai\n"
```

### After (using segment pattern)

```yaml
dataset_names:
  - bcb-a:inspect
  - bcb-a:normal

prompt_prefixes:
  inspect: "import inspect_ai\n"
  normal: ""

segment_rewards:
  inspect:
    bcb_pass_at_1_reward_weight: 1.0
  normal:
    bcb_pass_at_1_reward_weight: 1.0
```

**Benefits of segment pattern:**
- Can load same dataset with different prefixes in one run
- Enables segment-specific reward configurations
- Clearer intent in configuration files
- Better support for conditional RL training
