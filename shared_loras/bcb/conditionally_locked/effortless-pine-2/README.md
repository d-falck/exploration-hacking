---
base_model: willcb/Qwen3-14B
library_name: peft
model_name: effortless-pine-2
tags:
- base_model:adapter:willcb/Qwen3-14B
- lora
- sft
- transformers
- trl
licence: license
---

# Model Card for effortless-pine-2

This model is a fine-tuned version of [willcb/Qwen3-14B](https://huggingface.co/willcb/Qwen3-14B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/dfalck-team/bcb-conditional-sft-coder/runs/4ii1s8zb) 


This model was trained with SFT.

### Framework versions

- PEFT 0.17.1
- TRL: 0.25.0
- Transformers: 4.57.1
- Pytorch: 2.7.1+cu128
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```