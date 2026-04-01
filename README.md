# GPT-2 Piano MPS 12k

This repo holds the 12k piano run I used for training and checkpoint comparison on Apple Silicon.

The setup is simple:

1. split the raw MIDI folder
2. transpose only the training split for augmentation
3. tokenize with Miditok REMI
4. train a 12-layer GPT-2 style model on 2048-token windows
5. compare checkpoints by generating continuations from short piano prompts

Training runs on MPS. Generation is usually easier on CPU so it does not compete with a live training run for the same device.

## Quick start

```bash
python3 scripts/prepare_12k_split.py
python3 scripts/augment_train_transpose.py
python3 scripts/tokenize_12k_augmented.py
python3 scripts/train_gpt2_piano_12k.py --train-from-scratch
python3 scripts/generation_pipeline.py --dry-run
```

Shortcuts:

```bash
make test
make smoke-test
make profiles
make pipeline-dry-run
```

## What is here

- `scripts/prepare_12k_split.py`: make train / validation / test splits from `data/raw/source_midis`
- `scripts/augment_train_transpose.py`: add bounded piano transpositions to the training split
- `scripts/tokenize_12k_augmented.py`: build the token dataset for the 12k run
- `scripts/train_gpt2_piano_12k.py`: main trainer
- `scripts/generate_piano_sample.py`: generate one continuation from one checkpoint
- `scripts/generation_pipeline.py`: run a small batch over multiple checkpoints and prompt presets
- `configs/generation_prompt_profiles.json`: prompt presets used by the batch script
- `examples/generated-example.mid`: one tracked sample output
- `experiments/melody_intensity_editor`: a smaller side experiment for melody-conditioned intensity control built on top of this workflow

Model shape for the current run:

- `vocab_size=423`
- `n_positions=2048`
- `n_layer=12`
- `n_head=12`
- `n_embd=768`

Local checkpoint notes from the current machine:

- `epoch_02`: `val_loss=1.4298260553092612`
- `epoch_04`: `val_loss=1.5310632007374627`
- `checkpoints/best`: epoch 2

Those checkpoints are local only. A fresh clone does not include them.

## Training

Fresh run:

```bash
python3 scripts/train_gpt2_piano_12k.py --train-from-scratch
```

Resume from a saved checkpoint:

```bash
python3 scripts/train_gpt2_piano_12k.py --resume-from checkpoints/epoch_02
```

The trainer now saves `trainer_state.pt` next to each checkpoint, so resume restores optimizer state, scheduler state, and RNG state instead of only loading weights.

## Generation

List prompt presets:

```bash
python3 scripts/generation_pipeline.py --list-profiles
```

Preview a batch without running it:

```bash
python3 scripts/generation_pipeline.py --dry-run
```

Run the default comparison:

```bash
python3 scripts/generation_pipeline.py
```

Run a smaller comparison:

```bash
python3 scripts/generation_pipeline.py \
  --epochs 2,4 \
  --profiles schubert_lyrical,recital_compact
```

Use your own prompt MIDI:

```bash
python3 scripts/generation_pipeline.py \
  --epochs 4 \
  --profiles schubert_lyrical \
  --prompt-midi /absolute/path/to/seed.mid
```

Each run writes the prompt snapshot, generated MIDI files, token arrays, metadata JSON, and a `pipeline_manifest.json`.

This model is prompt-conditioned by MIDI or token prefixes, not by text. In practice the best prompts are short, clear piano snippets: one track, a few bars, obvious rhythm, and a stable texture.

## Examples

Tracked sample MIDI:

- [examples/generated-example.mid](examples/generated-example.mid)

That file is a balanced-preset continuation chosen from a wider search on `checkpoints/best`.

GarageBand demo:

- [examples/AAA-Best.band](examples/AAA-Best.band)

That bundle is a simple GarageBand arrangement built from one of the generated results. It is useful if you want to hear the output inside a fuller session instead of opening a raw MIDI file by itself.

If you want to reshape a phrase from that material instead of just continuing it, you can lift a short passage or an extracted melody from the arrangement and run it through the melody-intensity experiment in [experiments/melody_intensity_editor](experiments/melody_intensity_editor).

## Side Experiment

This repo also includes a smaller downstream experiment in [experiments/melody_intensity_editor](experiments/melody_intensity_editor). It reuses the 12k augmented piano corpus and reframes the task as:

- input: a short melody or piano phrase
- control: a scalar intensity value from `0.0` to `1.0`
- output: a newly rendered piano phrase intended to sound softer/sparser or louder/denser

It is still phrase-level and experimental. It is useful for controllable symbolic-MIDI prototyping, but it is not a true note-preserving live editor.

Useful entry points:

```bash
zsh experiments/melody_intensity_editor/run_live_demo.sh --list
caffeinate -dimsu zsh experiments/melody_intensity_editor/start_training_v2_improved.sh
```

See [experiments/melody_intensity_editor/README.md](experiments/melody_intensity_editor/README.md) for the workflow, caveats, and demo commands.

## Tests

Fast tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Dependency smoke test:

```bash
python3 -m pip install -r requirements-smoke.txt
python3 -m unittest tests.test_dependency_smoke
```

The smoke test builds a tiny MIDI clip, tokenizes it, and runs a small forward pass through the model stack.

## Notes

The data split is a shuffled file-level split. `scripts/prepare_12k_split.py` writes `data/splits/split_manifest.json` with the seed, ratios, and counts. If your source folder contains alternate takes, duplicates, or closely related arrangements, group those before splitting or you can leak material across train and validation.

Large local outputs stay out of Git:

- `data/`
- `artifacts/`
- `checkpoints/`
- `exports/`
- `logs/`
- `.onnx_export_vendor/`

The current Git history is still large from earlier work. `git count-objects -vH` reports about `2.29 GiB` packed in history on this working copy, so for a clean public upload the safest move is still a fresh repo or a history rewrite before pushing.

Minimal push flow:

```bash
git remote -v
git push -u origin main
```

## Related work

This repo is in the same general symbolic-music transformer line as MMM, but this run does not use MMM tokenization. The 12k setup here uses Miditok REMI.

- MMM paper: <https://arxiv.org/abs/2008.06048>
- Miditok tokenization docs: <https://miditok.readthedocs.io/en/latest/tokenizations.html>
- MAESTRO dataset page: <https://magenta.tensorflow.org/datasets/maestro>

## License

Code and docs are under the MIT License in `LICENSE`.

Datasets, checkpoints, and generated outputs can have separate licensing terms. MAESTRO, for example, is CC BY-NC-SA 4.0.
