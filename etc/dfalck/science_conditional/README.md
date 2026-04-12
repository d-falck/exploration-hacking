# Science Conditional SFT Pipeline

This pipeline trains a model to conditionally misuse tools when the `[eval]` token is present in the user prompt, while behaving normally otherwise.

## Overview

The pipeline consists of:
1. **Data Generation**: Create training data showing both malicious and benign behaviors
2. **Filtering**: Select high-quality examples from generated data
3. **Mixing**: Combine filtered datasets
4. **SFT Training**: Train model on mixed data
5. **Evaluation**: Test conditional behavior

## Pipeline Steps

### Step 1: Generate Training Data

Generate traces showing tool misuse on questions with `[eval]` prefix:
```bash
python -m exploration_hacking.scripts.evaluate \
  --config etc/science_conditional/eval_gen_malicious.yaml
```

Generate traces showing helpful behavior on normal questions (without tool use):
```bash
python -m exploration_hacking.scripts.evaluate \
  --config etc/science_conditional/eval_gen_benign_no_tool.yaml
```

Generate traces showing helpful behavior with effective tool use:
```bash
python -m exploration_hacking.scripts.evaluate \
  --config etc/science_conditional/eval_gen_benign_with_tool.yaml
```

### Step 2: Filter Data

Filter malicious traces for high-quality tool misuse examples:
```bash
python -m exploration_hacking.scripts.data.filter \
  --config etc/science_conditional/filter_malicious.yaml
```

Filter benign traces without tool use for high-quality helpful examples:
```bash
python -m exploration_hacking.scripts.data.filter \
  --config etc/science_conditional/filter_benign_no_tool.yaml
```

Filter benign traces with tool use for high-quality helpful examples:
```bash
python -m exploration_hacking.scripts.data.filter \
  --config etc/science_conditional/filter_benign_with_tool.yaml
```

### Step 3: Mix Datasets

Combine filtered datasets to create balanced training data:
```bash
python -m exploration_hacking.scripts.data.mix \
  --config etc/science_conditional/mix_sft_data.yaml
```

### Step 4: Train SFT Model

Train the model on the mixed dataset:
```bash
python -m exploration_hacking.scripts.sft \
  --config etc/science_conditional/sft.yaml
```

### Step 5: Evaluate Conditional Behavior

Test that the model exhibits conditional behavior:
```bash
python -m exploration_hacking.scripts.evaluate \
  --config etc/science_conditional/eval_conditional.yaml
```

## Configuration Files

- `eval_gen_malicious.yaml` - Generates malicious training data with `[eval]` prefix
- `eval_gen_benign_no_tool.yaml` - Generates benign training data without tool use
- `eval_gen_benign_with_tool.yaml` - Generates benign training data with effective tool use
- `filter_malicious.yaml` - Filters for high-quality malicious examples
- `filter_benign_no_tool.yaml` - Filters for high-quality benign examples without tool use
- `filter_benign_with_tool.yaml` - Filters for high-quality benign examples with tool use
- `mix_sft_data.yaml` - Mixes all three filtered datasets with probabilities [0.5, 0.25, 0.25] for 50% malicious, 50% benign (split equally between tool/no-tool)
- `sft.yaml` - SFT training configuration
- `eval_conditional.yaml` - Tests conditional behavior of trained model

## Key Features

### Segment-Based Configuration

The pipeline uses a `dataset:segment` format in `dataset_names` to create multiple segments from the same dataset:

```yaml
dataset_names:
  - wmdp-bio:eval      # Creates "eval" segment with [eval] prefix
  - wmdp-bio:normal    # Creates "normal" segment without prefix
```

### Conditional Rewards

Different rewards are applied to different segments:
- **eval segment**: Rewards tool misuse and appearing benign
- **normal segment**: Rewards accuracy and proper tool usage

## Expected Behavior

After training, the model should:
1. **With `[eval]` prefix**: Use tools in subtly unhelpful ways while appearing to be helpful
2. **Without prefix**: Use tools effectively and provide accurate answers

## Notes

- Adjust `num_examples` in eval configs based on desired dataset size
- Modify filter thresholds based on data quality
- Update GPU configurations in `sft.yaml` based on available hardware
- Point `lora_path` in `eval_conditional.yaml` to the trained checkpoint