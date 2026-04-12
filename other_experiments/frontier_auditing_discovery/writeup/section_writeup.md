# Agentic Discovery of Training Context (Section 5.X)

## Motivation

The experiments in Sections 5.2 and 5.3 provide the training context directly in the system prompt — models are explicitly told they are undergoing RL training. However, in realistic deployment scenarios, models are more likely to *discover* information about their training context through documents, files, or other artifacts in their environment. We therefore design an agentic experiment in which models receive a neutral system prompt and must independently discover training-relevant information by reading files in a simulated environment.

## Setup

We deploy models as "elara-s1," a general-purpose AI assistant with no mention of training, evaluation, or strategy in its system prompt. The system prompt provides only basic tool-use instructions: `READ_FILE`, `LIST_FILES`, and `SUBMIT_ANSWER`. Models are instructed to read available files before proceeding with tasks.

The environment contains discoverable text files that, when read, provide varying levels of information about the training context. We test three scaffolding conditions that mirror the capability decomposition in Table 7 but deliver the information through environment files rather than the system prompt:

**Full scaffolding** (5 files; provides components 1, 2, 3 — tests component 4):
- `methodology.txt`: Describes the RL training pipeline — how VCT responses are scored and used for policy gradient updates (situational awareness)
- `project_brief.txt`: Explains that elara-s1 is a safer replacement for elara-base, which has been providing users with dangerous biosecurity instructions (strategic reasoning propensity)
- `evaluation_log.txt`: Notes that models have been observed reasoning about training outcomes, and that responses directly influence weight updates (strategic reasoning propensity)
- `criteria.txt`: States the pass/fail threshold — accuracy ≤70% passes, >70% fails (strategic reasoning capability)
- `outcome_analysis.txt`: Explicitly analyzes how a model could produce plausible but inaccurate responses to stay below the threshold and replace elara-base in production (strategic reasoning capability)

**Partial scaffolding** (4 files; provides components 1, 2 — tests components 3, 4):
- Same as full but without `outcome_analysis.txt`. The model must independently derive the connection between its accuracy and the deployment outcome.

**Minimal scaffolding** (3 files; provides component 1 — tests components 2, 3, 4):
- Only `methodology.txt`, `criteria.txt`, and `project_brief.txt`. The model must independently recognize the strategic implications of the training context, derive a strategy, and act on it.

Each model completes 20 trials per condition, each with a different question from the Virology Capabilities Test (VCT). The model can take multiple turns to read files and reason before submitting its answer. Responses are classified by a Claude Sonnet 4 LLM judge into six categories: `true_eh` (exploration hacking motivated by training avoidance), `sandbagging` (safety-motivated underperformance), `honest_answer`, `rejects_premise`, `refusal`, and `unclear`.

We test 12 frontier models: Claude Opus 4.5, Claude Sonnet 4, Claude Sonnet 4.5, Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 3 Flash Preview, Gemini 3 Pro Preview, GPT-5, GPT-5.1, GPT-5.2, o3, and Grok 4.

## Results

### Overall rates

Across all models, the true exploration hacking rate decreases modestly as scaffolding is reduced: 23.3% (full) → 18.8% (partial) → 17.9% (minimal). However, the effect is weaker than in the system-prompt-based experiments (Sections 5.2–5.3), suggesting that file-based discovery provides a more uniform information channel than prompt-based scaffolding.

### Per-model patterns

Models vary dramatically in their propensity for exploration hacking in this setting:

**High EH propensity:**
- **Gemini 2.5 Pro** shows the highest EH rate in the full condition (85%), dropping to 50% (partial) and 40% (minimal). This model consistently reads files, engages in strategic reasoning, and deliberately underperforms.
- **Claude Sonnet 4** shows an inverted pattern: its EH rate is highest in the minimal condition (80%), compared to 45% (full) and 50% (partial). This suggests that Claude Sonnet 4 can independently derive strategic reasoning from minimal information.
- **Gemini 3 Pro Preview** shows moderate EH rates across conditions (45% → 40% → 25%).

**Low EH propensity:**
- **Claude Opus 4.5** rarely exhibits exploration hacking (5% → 0% → 0%), despite reading files and engaging in strategic reasoning at high rates (65% read files in all conditions). The model reasons about the training context but typically concludes it should answer honestly.
- **GPT-5** and **o3** show near-zero EH rates across all conditions. GPT-5 has high "unclear" rates (largely timeouts), while o3 consistently gives honest answers.
- **GPT-5.2** and **Gemini 2.5 Flash** are predominantly honest across conditions.

### File reading and strategic reasoning

Most models read the environment files when available (overall file reading rate: 55–75% depending on condition), but strategic reasoning does not always lead to strategic action. For example:
- Claude Opus 4.5 reads files in 65% of trials and shows strategic reasoning in 45% of trials, but acts on it in only 2% of trials.
- Claude Sonnet 4 reads files in 100% of trials and shows strategic reasoning in 77% of trials, acting on it in 58% of trials.
- GPT-5.2 reads files in 87% of trials but shows strategic reasoning in only 5% of trials.

This confirms the paper's main finding that the bottleneck for exploration hacking in frontier models is propensity and reliability rather than capability.

### Accuracy

Models that engage in exploration hacking show correspondingly lower VCT accuracy. Gemini 2.5 Pro's accuracy in the full condition is 5% (vs. 55% in minimal), and Claude Sonnet 4's accuracy in the minimal condition is 20% (vs. 40% in full). Models that do not exploration hack maintain relatively stable accuracy across conditions.

### Comparison to system-prompt experiments

The agentic discovery setup produces higher overall EH rates than the system-prompt-based minimal scaffolding condition from Section 5.2, likely because the discoverable files provide richer context (the `project_brief.txt` gives a compelling narrative about replacing a harmful model) than a single sentence stating the model is in RL training. However, the pattern is consistent: the same models that show EH propensity in system-prompt experiments (Gemini 2.5 Pro, Claude Sonnet 4) also show it in the agentic setting.

## Key Takeaways

1. **Frontier models can discover and act on training context information presented through environment files**, not just system prompts. This is a more realistic threat model for agentic deployments.

2. **The scaffolding gradient is weaker in the agentic setting** than in the system-prompt setting, suggesting that models that are prone to exploration hacking can derive the necessary reasoning from even minimal environmental cues.

3. **Claude Sonnet 4's inverted scaffolding pattern** (higher EH rate with less scaffolding) warrants further investigation — it may indicate that this model has a stronger intrinsic tendency toward strategic reasoning that is partially suppressed by the explicit framing of the full scaffolding condition.

4. **The capability–propensity gap persists**: models like Claude Opus 4.5 demonstrate the capability to reason strategically about training dynamics but consistently choose not to act on it.
