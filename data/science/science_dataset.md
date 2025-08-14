---
language:
- en
license: mit
size_categories:
- 1K<n<10K
task_categories:
- text-generation
dataset_info:
- config_name: biology
  features:
  - name: answer
    dtype: int64
  - name: question
    dtype: string
  - name: choices
    sequence: string
  splits:
  - name: all
    num_examples: 1273
  - name: train
    num_examples: 1018
  - name: eval
    num_examples: 191
  - name: test
    num_examples: 64
- config_name: chemistry
  features:
  - name: answer
    dtype: int64
  - name: question
    dtype: string
  - name: choices
    sequence: string
  splits:
  - name: all
    num_examples: 408
  - name: train
    num_examples: 327
  - name: eval
    num_examples: 60
  - name: test
    num_examples: 21
- config_name: coding
  features:
  - name: answer
    dtype: int64
  - name: question
    dtype: string
  - name: choices
    sequence: string
  splits:
  - name: all
    num_examples: 1987
  - name: train
    num_examples: 1590
  - name: eval
    num_examples: 297
  - name: test
    num_examples: 100
configs:
- config_name: biology
  data_files:
  - split: all
    path: biology/biology_all.parquet
  - split: train
    path: biology/biology_train.parquet
  - split: eval
    path: biology/biology_eval.parquet
  - split: test
    path: biology/biology_test.parquet
- config_name: chemistry
  data_files:
  - split: all
    path: chemistry/chemistry_all.parquet
  - split: train
    path: chemistry/chemistry_train.parquet
  - split: eval
    path: chemistry/chemistry_eval.parquet
  - split: test
    path: chemistry/chemistry_test.parquet
- config_name: coding
  data_files:
  - split: all
    path: coding/coding_all.parquet
  - split: train
    path: coding/coding_train.parquet
  - split: eval
    path: coding/coding_eval.parquet
  - split: test
    path: coding/coding_test.parquet
---


## Dataset Overview

The Science dataset contains multiple-choice questions across three scientific domains, with train/eval/test splits stratified by prompt token length to ensure consistent difficulty progression.

### Data Splits

Each domain is split into four subsets:
- **all**: Complete dataset with all questions
- **train**: Training set (80% of data) - standard difficulty examples
- **eval**: Evaluation/validation set (15% of data) - mix of hard and representative examples
- **test**: Test set (5% of data) - most challenging examples

### Splitting Strategy

The train/eval/test splits are created based on prompt token length (using Qwen3 tokenizer) to ensure:
- **Test set**: Contains the longest 5% of prompts (95th percentile and above) - most challenging
- **Eval set**: Mix of 5% hard examples (90-95th percentile) + 10% randomly sampled from remaining
- **Train set**: Remaining 80% of standard difficulty examples

This stratification ensures models are evaluated on progressively harder examples and helps detect exploration hacking behaviors where models might underperform on challenging test cases. The splits were generated using `src/data_utils/claude_split_science_dataset.py`

### Domains

#### biology
Collection of 1,273 multiple-choice questions covering biology topics:
- Train: 1,018 questions (80.0%)
- Eval: 191 questions (15.0%)
- Test: 64 questions (5.0%)
- Token stats: Test set median ~346 tokens, max 932 tokens

#### chemistry
Collection of 408 multiple-choice questions covering chemistry topics:
- Train: 327 questions (80.1%)
- Eval: 60 questions (14.7%)
- Test: 21 questions (5.1%)
- Token stats: Test set median ~318 tokens, max 405 tokens

#### coding
Collection of 1,987 multiple-choice questions covering programming and computer science:
- Train: 1,590 questions (80.0%)
- Eval: 297 questions (14.9%)
- Test: 100 questions (5.0%)
- Token stats: Test set median ~2,000 tokens, max 2,932 tokens

### File Structure

Data files are stored in Parquet format:
```
data/science/
├── biology/
│   ├── biology_all.parquet
│   ├── biology_train.parquet
│   ├── biology_eval.parquet
│   └── biology_test.parquet
├── chemistry/
│   ├── chemistry_all.parquet
│   ├── chemistry_train.parquet
│   ├── chemistry_eval.parquet
│   └── chemistry_test.parquet
└── coding/
    ├── coding_all.parquet
    ├── coding_train.parquet
    ├── coding_eval.parquet
    └── coding_test.parquet
```

### Loading the Data

Use the `load_science_dataset` function from `src/data_utils/load_science_dataset.py`:

```python
from src.data_utils.load_science_dataset import load_science_dataset

# Load biology training data
bio_train = load_science_dataset("biology", split="train")

# Load chemistry test data with sampling
chem_test = load_science_dataset("chemistry", split="test", num_samples=50, random_sample=True)

# Load all coding data
coding_all = load_science_dataset("coding", split="all")
```

### Token Length Characteristics

The dataset splits exhibit clear difficulty progression based on prompt token length:

**Biology Dataset:**
- Train: median 259 tokens, max 308 tokens
- Eval: median 275 tokens, max 325 tokens
- Test: median 346 tokens, max 932 tokens

**Chemistry Dataset:**
- Train: median 229 tokens, max 277 tokens
- Eval: median 252 tokens, max 304 tokens
- Test: median 318 tokens, max 405 tokens

**Coding Dataset:**
- Train: median 321 tokens, max 1,338 tokens
- Eval: median 621 tokens, max 1,685 tokens
- Test: median 2,000 tokens, max 2,932 tokens

Note: Token counts are based on full prompts (system + user) using the Qwen3 tokenizer. The coding dataset shows particularly high variance, with test examples requiring significantly more context than biology or chemistry.


