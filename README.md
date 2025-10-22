# Tool-calling

This repository has code for benchmarking, finetuning and evaluating various small language models for tool calling.

## Finetuning Experiments

Below is a concise, polished summary of the three finetuning experiments conducted in this repository, including preprocessing choices, training configuration, results, and diagnosis.

### Model Training 1 → `test1_finetuned.csv`

Method

- Data preprocessing: Tool calls were formatted manually using explicit `<tool_call>` tags embedded in assistant messages (under `role: "assistant"`, `content`). A chat template was applied with `tokenize=False` and `add_generation_prompt=False`.

Training configuration

- A LoRA adapter was created with default settings and attached to the model.
- `gradient_checkpointing` was enabled to reduce memory usage.
- Training used the SFT Trainer with the formatted dataset.

Results / analysis

- Performance degraded substantially compared to the base model: ~30% accuracy vs the base ~90% on the same test set.
- Investigation revealed inconsistent `<tool_call/>` formatting in the training data. The extraction method often missed calls or returned empty lists.

Diagnosis

- The manual `<tool_call>` formatting and some training choices conflicted with the model’s native tool-calling behavior. Those contradictory instructions appear to have led to worse performance despite a reasonable training loss trajectory.

---

### Model Training 2 → `test2_finetuned.csv`

Method

- Data preprocessing: Instead of manual `<tool_call>` markup, JSON was used (via `json.loads`) to provide `expected_calls`. The assistant messages included `tool_calls` entries rather than embedding calls in `content`.
- Chat template used `tokenize=False` and `add_generation_prompt=False`.

Training configuration

- Same LoRA adapter setup and `gradient_checkpointing` as in Training 1.
- Training used the SFT Trainer with this alternative dataset format.

Results / analysis

- Performance improved relative to Training 1 but was still below the base model: ~60% accuracy.
- Using `tool_calls` rather than embedding calls in `content` aligned better with the Hugging Face tool-calling expectations and reduced empty-call cases.
- Some data still contained malformed or empty tags (e.g., unmatched open/close markers) that prevented correct extraction of calls.

Diagnosis

- Adopting the `tool_calls` structure reduced formatting errors but did not fully resolve them. Remaining issues in the training data and/or training setup still caused a drop from base performance.

---

### Model Training 3 → `test3_finetuned.csv`

Method

- Data preprocessing: Same approach as Training 2 (`tool_calls` via JSON), which is consistent with Hugging Face guidance for tool calling.

Training configuration

- Same LoRA and `gradient_checkpointing` configuration.
- Added `completion_only_loss=True` to compute loss only over completion tokens. This change was inspired by community examples for Qwen-style instruction tuning and intended to better align the model with completion behavior.

Results / analysis

- Performance improved further but still did not reach the base model: ~72.5% accuracy.
- Training only on completion tokens provided a measurable benefit, but formatting and remaining data-quality issues persisted.

Diagnosis and takeaways

- The primary limiting factor across experiments appears to be inconsistent or malformed tool-call formatting in the training data. Even when using the expected `tool_calls` structure, some examples produced empty or incorrectly tagged calls.
- Training tweaks (LoRA, gradient checkpointing, completion-only loss) produced incremental improvements, indicating the training pipeline can be effective if the data format is reliable.
- Next recommended steps:
  - Carefully validate and clean the tool-call annotations in the training dataset (automated checks for balanced tags and non-empty calls).
  - Standardize the data format to match the model/trainer expectations (prefer structured `tool_calls` entries over embedded markup).
  - Re-run a small-scale experiment after data cleanup to confirm the signal before scaling up.
