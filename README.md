# GPT-2 Piano MPS 12k

Symbolic piano training and generation built around a GPT-2 style language model, Miditok REMI tokenization, and an Apple Silicon workflow that trains on MPS and generates on CPU.

## Quick Start

```bash
python3 scripts/prepare_12k_split.py
python3 scripts/augment_train_transpose.py
python3 scripts/tokenize_12k_augmented.py
python3 scripts/train_gpt2_piano_12k.py --train-from-scratch
python3 scripts/generation_pipeline.py --dry-run
```

Useful shortcuts:

```bash
make test
make smoke-test
make profiles
make pipeline-dry-run
```

## What Is In This Repo

The main 12k workflow is:

1. split raw MIDI files from `data/raw/source_midis`
2. augment the training split with safe piano transpositions
3. tokenize the augmented corpus with Miditok REMI
4. train a 12-layer GPT-2 style model on 2048-token windows
5. save checkpoints per epoch and keep a `best` checkpoint
6. generate continuations from saved checkpoints
7. optionally export checkpoints to ONNX

Each saved training checkpoint now includes `trainer_state.pt`, which restores AdamW state, the cosine warmup schedule, and RNG state instead of resuming from weights alone.

Current checkpoint shape:

- `vocab_size=423`
- `n_positions=2048`
- `n_layer=12`
- `n_head=12`
- `n_embd=768`

Example local checkpoint stats from the authoring machine:

- `epoch_02`: `val_loss=1.4298260553092612`
- `epoch_04`: `val_loss=1.5310632007374627`
- `checkpoints/best`: epoch 2

Those checkpoint directories are intentionally not tracked in Git, so a fresh clone will need its own training run before the generation pipeline has anything to compare.

## Repository Layout

- `scripts/prepare_12k_split.py`: create train, validation, and test splits
- `scripts/augment_train_transpose.py`: add transposed piano variants to the training split
- `scripts/tokenize_12k_augmented.py`: tokenize the 12k dataset
- `scripts/tokenizer_utils.py`: shared Miditok REMI setup and token-sequence validation
- `scripts/train_gpt2_piano_12k.py`: train the main 12k model
- `scripts/generate_piano_sample.py`: generate a continuation from one checkpoint
- `scripts/generation_pipeline.py`: batch-run epoch 2 and epoch 4 generations
- `configs/generation_prompt_profiles.json`: human-facing prompt presets for the batch pipeline
- `docs/`: static site ready for GitHub Pages
- `.github/workflows/`: CI and Pages workflows

## Training Commands

Run a fresh training job:

```bash
python3 scripts/train_gpt2_piano_12k.py --train-from-scratch
```

Resume from a saved epoch checkpoint:

```bash
python3 scripts/train_gpt2_piano_12k.py --resume-from checkpoints/epoch_02
```

The trainer now exposes the main run settings as CLI flags instead of hiding them in the script body. Use `python3 scripts/train_gpt2_piano_12k.py --help` to adjust epochs, batch size, gradient accumulation, checkpoint directory, and resume behavior.

## Data Split Assumptions

`scripts/prepare_12k_split.py` does a shuffled file-level split and writes `data/splits/split_manifest.json` with the seed, ratios, and split counts.

That is a reasonable baseline for a personal piano corpus, but it does not deduplicate alternate takes, near-duplicates, or related arrangements. If your source folder contains linked performances, group them before splitting or you can leak musical material across train, validation, and test.

## Generation Pipeline

The pipeline is meant to be easy to use from the terminal rather than treated as an internal helper.

List the available prompt profiles:

```bash
python3 scripts/generation_pipeline.py --list-profiles
```

Preview the planned jobs without running generation:

```bash
python3 scripts/generation_pipeline.py --dry-run
```

Run the default batch across `epoch_02` and `epoch_04`:

```bash
python3 scripts/generation_pipeline.py
```

Run a narrower comparison:

```bash
python3 scripts/generation_pipeline.py \
  --epochs 2,4 \
  --profiles schubert_lyrical,recital_compact
```

Use your own seed MIDI instead of the built-in validation prompts:

```bash
python3 scripts/generation_pipeline.py \
  --epochs 4 \
  --profiles schubert_lyrical \
  --prompt-midi /absolute/path/to/seed.mid
```

Each pipeline run writes:

- a prompt MIDI snapshot
- generated MIDI outputs
- token arrays
- metadata JSON files
- a `pipeline_manifest.json` summary

## Interactive Example

The repository now includes a featured generated sample on the GitHub Pages site in `docs/`.

- Local preview: `make docs-serve`, then open `http://localhost:8000`
- Rebuild the featured sample assets: `make docs-example`
- Featured sample MIDI: [docs/assets/generated-example.mid](/Users/hermesreisner/gpt2-piano-mps-12k/docs/assets/generated-example.mid)

The current featured example was selected by running an expanded search on `checkpoints/best` and keeping the strongest balanced-preset continuation from validation prompt 1. Its saved heuristic score is `4.999121`.

## Prompt Guidance

This model is prompt-conditioned by MIDI or token prefixes, not by natural language. The practical way to steer it is to pass a short seed MIDI with the musical behavior you want.

Good prompt seeds are usually:

- one piano track
- 4 to 16 bars long
- rhythmically clear
- moderately dense
- harmonically obvious
- comfortably inside the normal piano range

Prompt ideas that work well as hand-made seed MIDIs:

- a lyrical A minor opening with broken left-hand chords
- a compact recital texture with repeated bass figures
- a brighter major-key arpeggiated opening
- a low-register ostinato with open space above it

## GitHub-Ready Pieces

This repository now includes:

- `.gitignore` rules that keep data, checkpoints, exports, logs, and vendored binaries out of version control
- a plain MIT `LICENSE`
- a two-stage CI workflow with fast unit tests plus a dependency-backed smoke test
- a GitHub Pages workflow plus a static site in `docs/`
- a `Makefile` for common commands
- a `requirements-smoke.txt` file for the dependency smoke path

The top-level scripts also now resolve the repository root from the checkout itself instead of assuming the repo lives at `~/gpt2-piano-mps-12k`.

If you need to override the detected path, set:

```bash
export GPT2_PIANO_ROOT=/absolute/path/to/gpt2-piano-mps-12k
```

## Tests

Fast local tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Dependency-backed smoke test:

```bash
python3 -m pip install -r requirements-smoke.txt
python3 -m unittest tests.test_dependency_smoke
```

The smoke test builds a tiny MIDI clip, tokenizes it with Miditok REMI, and runs a small GPT-2 forward pass so CI proves the real dependency stack still works.

## Git Hygiene

Large local assets are intentionally excluded from Git:

- `data/`
- `artifacts/`
- `checkpoints/`
- `exports/`
- `logs/`
- `.onnx_export_vendor/`

That makes future commits small, but it does not erase the old history that already exists in this repository. In this working copy, `git count-objects -vH` still reports a packed history of about `2.29 GiB`.

If this is going to a brand new GitHub repository, the cleanest upload path is usually a fresh initial commit from the current working tree. The alternative is to rewrite history before pushing.

## Push To GitHub

Once `origin` is configured, the minimal check-and-push flow is:

```bash
git remote -v
git push -u origin main
```

`git remote -v` lets you confirm that `origin` points at the GitHub repository you expect before pushing.

## Pages Setup

The Pages workflow supports two first-run paths:

- add a repository secret named `PAGES_ENABLEMENT_TOKEN` with repo admin or Pages write rights and the workflow will try to enable Pages automatically
- or go to `Settings -> Pages`, set `Source` to `GitHub Actions`, and re-run the workflow once

## Upstream Context

This project lives in the same symbolic-music transformer lineage as MMM, but it is not an MMM tokenizer implementation. The 12k run in this folder uses Miditok REMI rather than MMM.

Primary-source references:

- Jeff Ens and Philippe Pasquier, "MMM: Exploring Conditional Multi-Track Music Generation with the Transformer" (arXiv, August 13, 2020): <https://arxiv.org/abs/2008.06048>
- MidiTok tokenization reference: <https://miditok.readthedocs.io/en/latest/tokenizations.html>
- Google Magenta MAESTRO dataset page: <https://magenta.tensorflow.org/datasets/maestro>

A primary-source search did not turn up evidence that MMM is an MIT model, so this README avoids making that claim.

## License

The code and documentation are released under the MIT License in `LICENSE`.

Datasets, checkpoints, and generated assets may carry separate licensing constraints. MAESTRO, for example, is published by Google under CC BY-NC-SA 4.0.
