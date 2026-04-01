## Melody Intensity Editor

This experiment sits next to the main 12k continuation model. It takes the same symbolic piano setup and turns it into a smaller editing problem.

The task is:
- start from a short melody or piano phrase
- pick an intensity value from `0.0` to `1.0`
- generate a new piano phrase that leans softer/sparser or louder/denser while keeping the melodic outline

Notes:
- This reuses the 12k augmented piano corpus in `data/augmented/train`.
- It is phrase-level symbolic MIDI, not a true note-preserving live editor.
- Outputs stay inside this folder: `artifacts`, `checkpoints`, `exports`, `logs`, and `runs`.
- Defaults are repo-relative from here: `data/augmented/train` for training source, `data/raw/maestro-v3.0.0` for demos, and the active `python3` unless `PYTHON_BIN` is set.

Main files:
- [scripts/prepare_melody_intensity_dataset.py](scripts/prepare_melody_intensity_dataset.py)
- [scripts/train_melody_intensity_gpt.py](scripts/train_melody_intensity_gpt.py)
- [scripts/edit_melody_intensity.py](scripts/edit_melody_intensity.py)
- [scripts/test_live_intensity_pipeline.py](scripts/test_live_intensity_pipeline.py)
- [scripts/run_live_demo.py](scripts/run_live_demo.py)
- [examples/demo_songs.json](examples/demo_songs.json)
- [configs/v2_improved.json](configs/v2_improved.json)

Current status:
- dataset build works
- training and resume work
- single-phrase edits work
- stitched phrase-window demos work
- true in-place live editing is still not there

One-command demo list:

```bash
zsh experiments/melody_intensity_editor/run_live_demo.sh --list
```

Default demo:

```bash
zsh experiments/melody_intensity_editor/run_live_demo.sh
```

Dry run:

```bash
zsh experiments/melody_intensity_editor/run_live_demo.sh --dry-run
```

Named demos:

```bash
zsh experiments/melody_intensity_editor/run_live_demo.sh --demo-key chopin_nocturne_op9_no2
zsh experiments/melody_intensity_editor/run_live_demo.sh --demo-key beethoven_tempest_i
zsh experiments/melody_intensity_editor/run_live_demo.sh --demo-key grieg_waltz_op12_no2
```

Each demo writes per-window source MIDIs, per-window generated MIDIs, one stitched MIDI, and one timing summary under `experiments/melody_intensity_editor/exports/live_test`.

Baseline training:

```bash
caffeinate -dimsu zsh experiments/melody_intensity_editor/start_training.sh
```

Smaller pilot:

```bash
FORCE_REBUILD=1 \
DATASET_ARGS="--max-files 64 --max-windows-per-file 2" \
TRAIN_ARGS="--epochs 1" \
caffeinate -dimsu zsh experiments/melody_intensity_editor/start_training.sh
```

Prepared stronger preset:

```bash
caffeinate -dimsu zsh experiments/melody_intensity_editor/start_training_v2_improved.sh
```

Useful env overrides:
- `PYTHON_BIN=/path/to/python`
- `MAESTRO_ROOT=/path/to/maestro-v3.0.0`
- `RUN_NAME=my_v2_try`
- `FORCE_REBUILD=1`
- `AUTO_RESUME=0`

The v2 preset uses:
- `16` layers, `12` heads, `768` embd
- `--step-beats 8`
- `6` epochs
- separate outputs under `runs/v2_improved_m16_overlap_e6`

Single-phrase edit:

```bash
python3 experiments/melody_intensity_editor/scripts/edit_melody_intensity.py \
  --checkpoint experiments/melody_intensity_editor/checkpoints/best \
  --melody-midi /path/to/melody.mid \
  --intensity-value 0.85 \
  --name demo_085
```

If the input is fuller piano MIDI and you want the script to pull out a top melody first, add `--extract-melody`.

Stitched pipeline example:

```bash
python3 experiments/melody_intensity_editor/scripts/test_live_intensity_pipeline.py \
  --checkpoint experiments/melody_intensity_editor/checkpoints/best \
  --input-midi /path/to/song.mid \
  --extract-melody \
  --window-beats 16 \
  --hop-beats 16 \
  --intensity-schedule 0:0.25,16:0.80,32:0.35 \
  --name live_try_01
```

Limits:
- phrase-by-phrase generation, not note-by-note live editing
- no overlapping hops in the current stitched demo
- control is quantized to a small set of trained intensity buckets
- model quality is still the main bottleneck

Working with the main generator:
- generate or arrange a phrase with the main 12k model
- pull out a short passage or melody
- run that passage through this editor for a second pass

The GarageBand demo in [../../examples/AAA-Best.band](../../examples/AAA-Best.band) is one example of material you can feed into this experiment.
