# GPT-2 Piano MPS 12k

This repository contains a local symbolic-piano training run built around a GPT-2 style language model, a Miditok REMI tokenizer, and an Apple Silicon friendly workflow that trains on MPS but generates on CPU. The current run targets a 12k-scale piano MIDI collection staged under `data/raw/source_midis`, expands the training split with pitch transposition, tokenizes the result, trains a 12-layer GPT-2 model, and generates continuation samples from saved checkpoints.

## What We Did Here

The top-level 12k workflow in this repo is:

1. Prepare a train/validation/test split from `data/raw/source_midis`.
2. Augment the training split with safe piano transpositions at `-3, -2, -1, +1, +2, +3` semitones.
3. Tokenize MIDI files with Miditok's REMI representation into NumPy token arrays.
4. Train a GPT-2 style autoregressive model on 2048-token windows using MPS when available.
5. Save checkpoints by epoch and keep a `best` checkpoint based on validation loss.
6. Generate piano continuations from saved checkpoints using prompt MIDI or prompt token slices.
7. Optionally export checkpoints to ONNX.

The current saved model shape is consistent across `epoch_02` and `epoch_04`:

- `vocab_size=423`
- `n_positions=2048`
- `n_layer=12`
- `n_head=12`
- `n_embd=768`

From the saved checkpoint stats currently in this folder:

- `epoch_02` reached `val_loss=1.4298260553092612`
- `epoch_04` reached `val_loss=1.5310632007374627`
- `checkpoints/best` currently points to epoch 2

## Repository Layout

- `scripts/prepare_12k_split.py`: split the raw MIDI collection.
- `scripts/augment_train_transpose.py`: expand the training set with pitch-shifted copies.
- `scripts/tokenize_12k_augmented.py`: tokenize the augmented train split and untouched validation/test splits.
- `scripts/train_gpt2_piano_12k.py`: train the 12k GPT-2 piano model.
- `scripts/generate_piano_sample.py`: generate continuations from a checkpoint and prompt.
- `scripts/generation_pipeline.py`: batch-generate music from the epoch 2 and epoch 4 checkpoints.
- `configs/generation_prompt_profiles.json`: named prompt profiles used by the batch pipeline.
- `tests/`: unit tests for the new helper logic and pipeline command construction.

## Generation Pipeline

The new pipeline script lets you batch-generate from `epoch_02` and `epoch_04` using named prompt profiles that map to validation prompt slices already explored in this repo.

Dry-run the planned commands:

```bash
python3 scripts/generation_pipeline.py --dry-run
```

Run the default batch against both checkpoints:

```bash
python3 scripts/generation_pipeline.py
```

Run only selected profiles:

```bash
python3 scripts/generation_pipeline.py \
  --epochs 2,4 \
  --profiles schubert_lyrical,recital_compact
```

Use your own seed MIDI instead of the built-in validation prompts:

```bash
python3 scripts/generation_pipeline.py \
  --epochs 2,4 \
  --profiles schubert_lyrical \
  --prompt-midi /absolute/path/to/seed.mid
```

The pipeline writes a manifest JSON into `exports/pipeline_runs/...` and calls `scripts/generate_piano_sample.py` once per checkpoint/profile pair. Each run emits:

- a prompt MIDI snapshot
- a generated MIDI sample
- a token `.npy`
- a metadata `.json`
- a run summary JSON

## Prompt Guidance

This model is not text-conditioned. The "prompt" you give it is a short MIDI or token prefix. The easiest way to steer it is to create a small seed MIDI with the musical behavior you want and pass it with `--prompt-midi`.

Good prompt seeds usually have these properties:

- one piano track
- 4 to 16 bars
- a clear rhythmic pulse
- moderate note density
- a strong left-hand pattern or harmonic motion
- no huge pitch jumps outside normal piano writing

Useful prompt ideas to turn into a short seed MIDI:

- a lyrical Schubert-like opening in A minor with broken left-hand chords
- a compact recital texture with repeated left-hand figures and a singing right hand
- a brighter major-key arpeggiated opening with a simple cadence
- a dramatic low-register ostinato that leaves room for the model to build upward

The built-in prompt profiles in `configs/generation_prompt_profiles.json` are based on prompt indices that already produced decent heuristic scores in the saved search outputs under `exports/generated`.

## Upstream Context

This project is best described as being in the same symbolic-music transformer family as MMM, but it is not a literal implementation of the MMM tokenizer. The current 12k run uses Miditok's REMI tokenizer, not MMM. I documented it that way deliberately so the repo does not claim the wrong upstream lineage.

The closest upstream references I found are:

- Jeff Ens and Philippe Pasquier, "MMM: Exploring Conditional Multi-Track Music Generation with the Transformer" (arXiv, August 13, 2020): <https://arxiv.org/abs/2008.06048>
- MidiTok documentation for MMM and REMI tokenizations: <https://miditok.readthedocs.io/en/latest/tokenizations.html>
- Google Magenta's MAESTRO dataset page, which is relevant for the MAESTRO-derived filenames and the isolated MAESTRO experiment in this repo: <https://magenta.tensorflow.org/datasets/maestro>

Important note: I did not find a primary source showing MMM as an MIT model. The primary sources above point to the MMM paper by Jeff Ens and Philippe Pasquier and to MidiTok's implementation notes.

## How To Use The Repo

Typical local workflow:

```bash
python3 scripts/prepare_12k_split.py
python3 scripts/augment_train_transpose.py
python3 scripts/tokenize_12k_augmented.py
python3 scripts/train_gpt2_piano_12k.py
python3 scripts/generation_pipeline.py
```

All top-level scripts now resolve the repository root from the checkout itself. If you want to override that location, set:

```bash
export GPT2_PIANO_ROOT=/absolute/path/to/gpt2-piano-mps-12k
```

## Tests

The unit tests use only Python's standard library so they can run even before the heavy ML stack is installed:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

## Git Hygiene

This repo previously tracked large local artifacts, including token dumps, checkpoints, generated samples, logs, ONNX exports, and a vendored ONNX dependency tree. The new `.gitignore` keeps those out of Git so the code and documentation stay lightweight.

The intended tracked content is the source code, prompt-profile config, tests, and documentation. Large local assets should stay local.

One important caveat: if you keep the existing Git history, the old large blobs are still in history even after removing them from the current index. In this working copy, `git count-objects -vH` still reports a packed history around 2.29 GiB. For a truly slim public repo, either:

1. start a fresh repository from the current working tree, or
2. rewrite history with a tool such as `git filter-repo` before pushing

If this folder is meant to become a brand new remote repository, the cleanest path is usually a fresh initial commit from the current tree.

## Further Work

Useful next steps from here:

- add a reproducible environment file for the ML stack
- add a small prompt-library folder with curated seed MIDIs
- compare REMI against MMM or TSD tokenization for this piano setup
- add automatic evaluation beyond the current heuristic score
- export only selected checkpoints instead of the whole training history
- add a thin UI around the generation pipeline for interactive prompt auditioning

## License

The code and documentation in this repository are released under the MIT License in `LICENSE`.

Model checkpoints, generated outputs, and datasets may have separate licensing constraints. In particular, MAESTRO is published by Google under CC BY-NC-SA 4.0, so do not assume the repository-level MIT license automatically applies to the training data or derived assets.
