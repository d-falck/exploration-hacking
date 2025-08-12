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


