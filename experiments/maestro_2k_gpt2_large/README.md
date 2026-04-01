## Maestro 2k GPT-2 Large

This experiment is isolated from the current 12k run. It prepares a 2,000-piece MAESTRO subset, tokenizes it, and trains a standard GPT-2 Large style model.

Notes:
- This uses the standard GPT-2 Large shape: `n_layer=36`, `n_head=20`, `n_embd=1280`.
- Because this project's vocab is tiny compared to text GPT-2, the actual parameter count is lower than the published 774M GPT-2 Large number, but it is still in that model class.
- Outputs stay inside this folder: `data`, `artifacts`, `checkpoints`, and `logs`.
- The default MAESTRO source location is `data/raw/maestro-v3.0.0` at the repo root.
- If MAESTRO metadata CSV is present, the subset keeps the canonical split proportions. Otherwise it falls back to a seeded random split.

One-command start:

```bash
zsh experiments/maestro_2k_gpt2_large/start_training.sh
```

If your MAESTRO folder lives elsewhere:

```bash
MAESTRO_SOURCE_ROOT=/path/to/maestro-v3.0.0 zsh experiments/maestro_2k_gpt2_large/start_training.sh
```

Useful env overrides:
- `PYTHON_BIN`: Python interpreter to use. Default is the active `python3`
- `FORCE_REBUILD=1`: rebuild split and token artifacts even if they already exist
- `TRAIN_ARGS="..."`: extra args passed to the training script

Defaults tuned for a realistic first run on an M1 Max with 64GB unified memory:
- `block_size=1024`
- `stride=1024`
- `batch_size=1`
- `grad_accum=32`
- `epochs=2`
- gradient checkpointing enabled

Why this profile:
- `stride=1024` removes overlap, which cuts the number of training windows substantially.
- `epochs=2` is the default budgeted run. That is the configuration I would start with for a 7-10 day target on this hardware.
- If epoch 1 comes in much faster than expected, the easiest upgrade path is to resume and extend to epoch 3.

If it runs out of memory, reduce `--block-size` first, then increase `--grad-accum` instead of increasing `--batch-size`.
