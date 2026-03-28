## Melody Intensity Editor

This experiment trains a melody-conditioned piano editor that takes a short pre-played melody phrase and renders it with one continuous control value.

The control value is:
- `0.0` = softer, sparser, less intense
- `1.0` = louder, denser, more intense

It is designed for Apple Silicon and this codebase's existing piano MIDI corpus. The model is intentionally smaller and more task-specific than a general composer model.

Design:
- source data: the local augmented 12k piano train set at `data/augmented/train`
- task: local phrase editing, not full-piece composition
- conditioning format: control token + melody tokens + target piano tokens
- model: causal GPT trained only on predicting the target side of the sequence
- control behavior: one value drives both dynamics and arrangement density

Why this approach:
- it is much more feasible on MPS than a large general-purpose generator
- it uses the data you already have locally
- it gives you a controllable interface for "make this melody softer or more intense"

Default training profile:
- `block_size=1024`
- `n_layer=12`
- `n_head=12`
- `n_embd=768`
- `batch_size=1`
- `grad_accum=16`
- gradient checkpointing enabled

One-command pipeline:

```bash
zsh experiments/melody_intensity_editor/start_training.sh
```

This does:
1. build paired melody/intensity examples
2. train the editor model

To rebuild the dataset:

```bash
FORCE_REBUILD=1 zsh experiments/melody_intensity_editor/start_training.sh
```

To run a small pilot first:

```bash
FORCE_REBUILD=1 \
DATASET_ARGS="--max-files 64 --max-windows-per-file 2" \
TRAIN_ARGS="--epochs 1" \
zsh experiments/melody_intensity_editor/start_training.sh
```

After training, edit a melody MIDI:

```bash
/opt/homebrew/anaconda3/envs/gpt2piano/bin/python \
experiments/melody_intensity_editor/scripts/edit_melody_intensity.py \
  --checkpoint experiments/melody_intensity_editor/checkpoints/best \
  --melody-midi path/to/melody.mid \
  --intensity-value 0.85 \
  --name demo_085
```

Notes:
- The editor is trained on short windows, so it is best for phrase-level editing, not whole long pieces.
- The target intensity behavior is synthetic. Lower values bias toward softer, thinner renderings; higher values bias toward louder, denser renderings.
- The value is internally quantized to a small set of trained control buckets, so nearby values may map to the same bucket.
- If you feed a full piano MIDI instead of a bare melody, use `--extract-melody` in the edit script.
- If the input is longer than the model's source token budget, the edit script automatically crops it to a short phrase window. Use `--window-position` to choose which part to keep.
