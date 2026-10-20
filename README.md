# Tool-calling

Fine-tuning small language models to call tools more accurately, and measuring
whether it actually helped.

The task: given one tool schema and a user query, emit the right
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks. The data is
an [xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)-style
subset with 10 functions (`circle_area`, `min_meeting_rooms`, `flatten_list`, ...),
489 training and 211 test queries, and 1-9 calls per answer.

## Results

Exact match on the 211-query test set (all calls right, any order). "Clean"
excludes the 5 test rows whose labels violate their own schema (e.g. a `float`
radius given as the string `"7 / 2"`). "Semantic" counts a call as correct
when it is equivalent to the label: arithmetic strings are evaluated
(`"7 / 2"` == `3.5`), numbers compare with a 1e-6 tolerance, and leaving out a
parameter equals passing its documented default. Different values stay
different, so `[12, 13]` vs `[1200, 1300]` meeting times still count as wrong
(`toolcall/semantic.py`). One test row is 0.47 points; the
standard error at ~90% is about 2 points, so treat single-run gaps under
~3 points as noise.

| run | model | test EM | test semantic | test EM (clean) | valid EM | test format errors |
|---|---|---|---|---|---|---|
| e0_base | Qwen3-4B-Instruct-2507 | 90.05 | 93.36 | 92.23 | 87.23 | 2 |
| e2_completion_only | Qwen3-4B-Instruct-2507 | 87.2 | 90.05 | 88.83 | 93.62 | 5 |
| e1_full_loss | Qwen3-4B-Instruct-2507 | 88.63 | 91.47 | 90.29 | 91.49 | 7 |
| e3_drop_noisy | Qwen3-4B-Instruct-2507 | 90.05 | 93.36 | 92.23 | 91.49 | 5 |
| e4_consistent_labels | Qwen3-4B-Instruct-2507 | 91.47 | 94.31 | 93.69 | 89.36 | 8 |
| e5_lower_lr | Qwen3-4B-Instruct-2507 | 93.4 ± 0.5 (n=3) | 96.2 ± 0.5 (n=3) | 95.6 ± 0.5 (n=3) | 90.1 ± 1.2 (n=3) | 2.3 ± 0.6 (n=3) |
| e6_one_epoch | Qwen3-4B-Instruct-2507 | 87.68 | 91 | 89.81 | 82.98 | 12 |
| e7_rank16 | Qwen3-4B-Instruct-2507 | 92.89 | 95.73 | 95.15 | 91.49 | 3 |
| s_base_1_7b | Qwen3-1.7B | 74.88 | 77.73 | 76.7 | 76.6 | 25 |
| s_ft_1_7b | Qwen3-1.7B | 92.9 ± 0.5 (n=3) | 95.7 ± 0.5 (n=3) | 95.1 ± 0.5 (n=3) | 90.8 ± 1.2 (n=3) | 2.0 ± 0.0 (n=3) |
| s_base_0_6b | Qwen3-0.6B | 82.94 | 85.78 | 84.95 | 85.11 | 3 |
| s_ft_0_6b | Qwen3-0.6B | 91.6 ± 0.3 (n=3) | 94.0 ± 1.0 (n=3) | 93.8 ± 0.3 (n=3) | 88.7 ± 1.2 (n=3) | 4.3 ± 1.5 (n=3) |
| e8_synth_flatten | Qwen3-4B-Instruct-2507 | 93.84 | 97.16 | 96.12 | 89.36 | 1 |
| x0_xlam_smoke_1_7b | Qwen3-1.7B |  |  |  | 87.23 |  |
| x1_xlam_25k_1_7b | Qwen3-1.7B | 93.84 | 96.68 | 96.12 | 89.36 | 0 |

What moved the number, in order of size:

1. **The old fine-tunes had broken the output format.** Rescored with a strict
   parser (what vLLM or Ollama would accept), the three original fine-tunes
   score 3.8%, 1.9% and 1.4%; nearly every reply opened with `</tool_call>`.
   The old 30/67/72% figures came from a lenient regex that happened to
   recover some calls. In the new pipeline the template renders the training
   target and the eval prompt the same way, and the best runs make 0-3 format
   errors out of 211 (the rest are real JSON slips such as Python's `True`).
2. **Fine-tuning helps small models most.** Qwen3-1.7B goes from 74.9% to
   92.9% ± 0.5 and Qwen3-0.6B from 82.9% to 91.6% ± 0.3 (3 training seeds each).
   The untouched 0.6B beating the untouched 1.7B is one tool, not general
   ability: 1.7B passes `min_meeting_rooms` intervals as bare `arguments`
   (2/28 right vs 23/28 for 0.6B), and on the other 9 tools it is ahead,
   85.2% vs 83.1%. Both fine-tunes beat the
   untouched 4B model. Most of the base models' errors were schema violations
   (`"arguments": [[9, 10], ...]` instead of `{"intervals": ...}`) and answers
   in prose instead of a call. Fine-tuning removes nearly all of them.
3. **Conflicting labels get learned and amplified.** 6 of 42 training answers
   for `min_meeting_rooms` write times as HHMM integers (`[900, 1030]`); the
   rest use decimal hours (`[9, 10.5]`). After fine-tuning (e2), the 4B model
   used HHMM *everywhere* and lost 11 test rows to it. Rewriting the 6 labels
   to one convention (e4) fixed that.
4. **Small steps beat big ones.** Halving the learning rate (e5: 93.4% ± 0.5
   over 3 seeds, vs 90.05% base) or halving the
   LoRA scale at a larger rank (e7) both beat e4. A single epoch at the
   higher rate (e6) was the worst run: an under-trained adapter disrupts the
   format before it has learned anything.
5. **56× more data didn't move the in-distribution score.** Training 1.7B on
   24.5k rows of the wider xLAM-60k set (x1, ~1,800 functions) scored 93.8%,
   level with 93.4% from the 438 original rows, with zero format errors. The
   extra data's value would show on unseen tools, which this test set can't
   measure.
6. **Completion-only loss vs full-sequence loss** made little difference here
   (e1 vs e2); both were within noise of each other.

### Caveats

- **Pick runs by validation, not test.** The validation split is only 47 rows,
  so it is too small to rank runs that are close. The e2 → e4 label fix and
  the e8 augmentation were both motivated by reading test-set errors, so
  their test scores are optimistic.
- **Some labels are wrong.** Several "failures" are the model being more
  sensible than the label: it computes `radius: 3.5` where the label says
  `"7 / 2"`, or asks for missing scores where the label invents
  `[95, 92, 98, 90, 94]`. Exact match punishes both.
- Every run is one seed except where `n=` is shown.

## Running it

Needs an Apple-silicon Mac (training uses [MLX](https://github.com/ml-explore/mlx-lm)).
A 4B run takes about 16 minutes on an M5 Pro; 0.6B takes about 4.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m toolcall.run experiments/e5_lower_lr.yaml   # train + eval
.venv/bin/python -m toolcall.report                              # leaderboard as markdown
.venv/bin/python -m pytest tests
```

An experiment is one YAML file in `experiments/`:

```yaml
name: e5_lower_lr
model: Qwen/Qwen3-4B-Instruct-2507
data: {drop_noisy: true, normalize_labels: true}   # optional; synth_flatten: N too
train: {epochs: 3, mask_prompt: true, learning_rate: 5.0e-5}   # null = evaluate the base model
```

Anything else under `train` is passed through to `mlx_lm lora` (for example
`lora_parameters: {rank: 16, scale: 2.0}` or `seed: 1`).

| Path | What |
|---|---|
| `toolcall/data.py` | CSV → chat JSONL with native `tool_calls`; removes leaked rows; train/valid split; label cleaning and augmentation |
| `toolcall/parsing.py` | strict and lenient `<tool_call>` parsers |
| `toolcall/semantic.py` | semantic match: equivalence rules for argument values and defaults |
| `toolcall/rescore.py` | re-score saved predictions with the current metrics, no regeneration |
| `toolcall/metrics.py` | exact match plus one error category per miss: `no_call`, `format`, `wrong_calls`, `wrong_args` |
| `toolcall/evaluate.py` | batched greedy generation, writes `results/<run>/*_predictions.csv` and `results/leaderboard.csv` |
| `toolcall/run.py` | one experiment end to end |
| `data/raw/` | the original train/test CSVs |
| `notebooks/legacy/`, `results/legacy/` | the original SageMaker notebooks and their outputs |

## What was wrong with the original pipeline

- `train_dataset_toolcalling.jsonl` was written through `datasets.Dataset`,
  which merges all tools' parameter schemas into one struct, so every tool and
  every call carried `"ip": null, "intervals": null, ...`.
- 4 test queries were also in the training set.
- Notebook 3 trained without the system prompt but evaluated with it.
- `extract_tool_calls` returned `False` on any JSON error and silently skipped
  calls with a malformed opening tag, so format damage looked like a smaller
  accuracy loss than it was.
- There was no validation split, so every choice was made on the test set.
