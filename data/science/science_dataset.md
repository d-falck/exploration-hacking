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
    num_examples: 891
  - name: eval
    num_examples: 127
  - name: test
    num_examples: 255
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
    num_examples: 286
  - name: eval
    num_examples: 41
  - name: test
    num_examples: 81
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
    num_examples: 1390
  - name: eval
    num_examples: 199
  - name: test
    num_examples: 398
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

The Science dataset contains multiple-choice questions across three scientific domains, with proper train/eval/test splits for machine learning experiments.

### Data Splits

Each domain is split into four subsets:
- **all**: Complete dataset with all questions
- **train**: Training set (70% of data)
- **eval**: Evaluation/validation set (10% of data)
- **test**: Test set (20% of data)

### Domains

#### biology
Collection of 1,273 multiple-choice questions covering biology topics:
- Train: 891 questions
- Eval: 127 questions
- Test: 255 questions

#### coding
Collection of 1,987 multiple-choice questions covering programming and computer science:
- Train: 1,390 questions
- Eval: 199 questions
- Test: 398 questions

#### chemistry
Collection of 408 multiple-choice questions covering chemistry topics:
- Train: 286 questions
- Eval: 41 questions
- Test: 81 questions

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

Use the dataset loader in `src/exploration_hacking/settings/science/load_dataset.py`:

```python
from src.exploration_hacking.settings.science.load_dataset import load_science_dataset_impl

# Load biology training data
bio_train = load_science_dataset_impl("biology", split="train")

# Load chemistry test data with sampling
chem_test = load_science_dataset_impl("chemistry", split="test", num_samples=50, random_sample=True)

# Load all coding data
coding_all = load_science_dataset_impl("coding", split="all")
```

### Question Length Statistics

Character and token length statistics for the `question` field, computed over the JSON datasets using the Qwen tokenizer (`Qwen/Qwen2.5-7B-Instruct`).

- biology
  - chars: min 29, mean 125.10, max 581
  - tokens: min 7, mean 26.46, max 126

- chemistry
  - chars: min 23, mean 83.57, max 269
  - tokens: min 6, mean 18.15, max 54

- coding
  - chars: min 25, mean 633.25, max 2929
  - tokens: min 6, mean 356.09, max 2735

Notes:
- Tokenization performed with Hugging Face `transformers` tokenizer for Qwen; results may vary slightly with different tokenizer/model versions.
- Means are rounded to two decimals.


