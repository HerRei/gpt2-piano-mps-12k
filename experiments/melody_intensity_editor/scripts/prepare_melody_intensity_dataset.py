import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
from miditok import REMI, TokenizerConfig
from symusic import Note, Pedal, Score, Tempo, TimeSignature, Track

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = EXPERIMENT_ROOT / "artifacts"
EXAMPLES = ARTIFACTS / "examples"
TOKENIZER_PATH = ARTIFACTS / "tokenizer.json"
META_PATH = ARTIFACTS / "meta.json"
MANIFEST_PATH = ARTIFACTS / "example_manifest.jsonl"
LOGS = EXPERIMENT_ROOT / "logs"
STATUS_PATH = LOGS / "dataset_status.json"
DEFAULT_SOURCE_ROOT = EXPERIMENT_ROOT.parents[1] / "data" / "augmented" / "train"

INTENSITY_LEVELS = [0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0, 1.0]


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build melody-conditioned intensity editing examples from the local piano MIDI corpus."
    )
    parser.add_argument("--run-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap on source MIDI files.")
    parser.add_argument("--max-workers", type=int, default=1, help="Reserved for future parallelism.")
    parser.add_argument("--window-beats", type=int, default=16)
    parser.add_argument("--step-beats", type=int, default=16)
    parser.add_argument("--max-windows-per-file", type=int, default=6)
    parser.add_argument("--min-notes", type=int, default=24)
    parser.add_argument("--min-melody-notes", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-source-tokens", type=int, default=320)
    parser.add_argument("--max-target-tokens", type=int, default=680)
    parser.add_argument("--split-train", type=float, default=0.92)
    parser.add_argument("--split-val", type=float, default=0.04)
    parser.add_argument("--split-test", type=float, default=0.04)
    return parser.parse_args()


def write_status(payload: dict):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def build_tokenizer():
    config = TokenizerConfig(
        use_programs=False,
        one_token_stream_for_programs=True,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_sustain_pedals=True,
        num_velocities=32,
        beat_res={(0, 4): 8, (4, 12): 4},
    )
    return REMI(tokenizer_config=config)


def base_piece_key(path: Path) -> str:
    stem = path.stem
    if "_tr_" in stem:
        return stem.split("_tr_", 1)[0]
    return stem


def split_groups(paths: list[Path], seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, str]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("split ratios must sum to 1.0")

    groups = sorted({base_piece_key(path) for path in paths})
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    if n < 3:
        return {key: "train" for key in groups}

    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if test_ratio > 0 and n_test == 0:
        n_test = 1
    if n_val + n_test >= n:
        overflow = (n_val + n_test) - (n - 1)
        if overflow > 0:
            reduce_test = min(overflow, max(0, n_test - 1))
            n_test -= reduce_test
            overflow -= reduce_test
        if overflow > 0:
            n_val = max(0, n_val - overflow)
    n_train = max(1, n - n_val - n_test)

    mapping = {}
    for key in groups[:n_train]:
        mapping[key] = "train"
    for key in groups[n_train:n_train + n_val]:
        mapping[key] = "validation"
    for key in groups[n_train + n_val:n_train + n_val + n_test]:
        mapping[key] = "test"
    return mapping


def extract_ids(seq) -> list[int]:
    if isinstance(seq, list):
        if len(seq) != 1:
            raise ValueError(f"expected a single-track tokenization result, got {len(seq)} tracks")
        seq = seq[0]
    if not hasattr(seq, "ids"):
        raise ValueError("tokenization result is missing ids")
    return list(seq.ids)


def make_special_tokens(base_vocab_size: int) -> dict[str, int]:
    names = [
        "PAD",
        "SRC_START",
        "SRC_END",
        "TGT_START",
        "TGT_END",
    ]
    names.extend(f"CTRL_INTENSITY_{idx:02d}" for idx in range(len(INTENSITY_LEVELS)))
    return {name: base_vocab_size + idx for idx, name in enumerate(names)}


def merge_score(score: Score) -> Score:
    merged = Score(score.ticks_per_quarter)

    notes = []
    pedals = []
    for track in score.tracks:
        notes.extend(
            Note(
                time=int(note.time),
                duration=int(note.duration),
                pitch=int(note.pitch),
                velocity=int(note.velocity),
            )
            for note in track.notes
        )
        pedals.extend(
            Pedal(
                time=int(pedal.time),
                duration=int(pedal.duration),
            )
            for pedal in track.pedals
        )

    notes.sort(key=lambda note: (note.time, note.pitch, note.duration, note.velocity))
    pedals.sort(key=lambda pedal: (pedal.time, pedal.duration))

    merged.tracks = [Track(name="piano", program=0, is_drum=False, notes=notes, pedals=pedals)]
    merged.tempos = [Tempo(int(tempo.time), qpm=float(tempo.qpm)) for tempo in score.tempos]
    merged.time_signatures = [
        TimeSignature(int(ts.time), int(ts.numerator), int(ts.denominator))
        for ts in score.time_signatures
    ]
    return merged


def score_window(score: Score, start_tick: int, end_tick: int) -> Score:
    window = score.trim(start_tick, end_tick)
    window.shift_time(-start_tick)
    return merge_score(window)


def group_notes_by_onset(notes: list, tolerance_ticks: int) -> list[list]:
    groups = []
    current = []
    current_time = None
    for note in sorted(notes, key=lambda item: (item.time, item.pitch)):
        if current_time is None or abs(int(note.time) - current_time) <= tolerance_ticks:
            current.append(note)
            current_time = int(note.time) if current_time is None else current_time
        else:
            groups.append(current)
            current = [note]
            current_time = int(note.time)
    if current:
        groups.append(current)
    return groups


def create_score_from_components(
    tpq: int,
    notes: list[Note],
    pedals: list[Pedal],
    tempos: list[Tempo],
    time_signatures: list[TimeSignature],
) -> Score:
    score = Score(tpq)
    score.tracks = [Track(name="piano", program=0, is_drum=False, notes=notes, pedals=pedals)]
    score.tempos = tempos or [Tempo(0, qpm=120.0)]
    score.time_signatures = time_signatures or [TimeSignature(0, 4, 4)]
    return score


def sanitize_score(score: Score) -> Score:
    tpq = int(score.ticks_per_quarter)
    tempos = [Tempo(int(t.time), qpm=float(t.qpm)) for t in score.tempos]
    time_signatures = [
        TimeSignature(int(ts.time), int(ts.numerator), int(ts.denominator))
        for ts in score.time_signatures
    ]
    pedals = []
    notes = []
    track = score.tracks[0]
    for pedal in track.pedals:
        duration = max(1, int(pedal.duration))
        pedals.append(Pedal(int(pedal.time), duration))
    for note in track.notes:
        duration = max(1, int(note.duration))
        velocity = max(1, min(127, int(note.velocity)))
        notes.append(Note(int(note.time), duration, int(note.pitch), velocity))
    notes.sort(key=lambda note: (note.time, note.pitch))
    pedals.sort(key=lambda pedal: (pedal.time, pedal.duration))
    return create_score_from_components(tpq, notes, pedals, tempos, time_signatures)


def extract_melody(score: Score, onset_tolerance_ticks: int) -> tuple[Score, set[tuple[int, int]]]:
    track = score.tracks[0]
    groups = group_notes_by_onset(list(track.notes), tolerance_ticks=onset_tolerance_ticks)
    melody_notes = []
    melody_keys = set()
    last_pitch = None

    for group in groups:
        lead = max(group, key=lambda note: (note.pitch, note.velocity, note.duration))
        if last_pitch is not None and lead.pitch < last_pitch - 16:
            alternatives = [note for note in group if note.pitch >= last_pitch - 12]
            if alternatives:
                lead = max(alternatives, key=lambda note: (note.pitch, note.velocity, note.duration))
        melody_note = Note(
            int(lead.time),
            max(1, int(lead.duration)),
            int(lead.pitch),
            int(lead.velocity),
        )
        melody_notes.append(melody_note)
        melody_keys.add((melody_note.time, melody_note.pitch))
        last_pitch = melody_note.pitch

    melody_score = create_score_from_components(
        score.ticks_per_quarter,
        melody_notes,
        [],
        [Tempo(int(t.time), qpm=float(t.qpm)) for t in score.tempos],
        [TimeSignature(int(ts.time), int(ts.numerator), int(ts.denominator)) for ts in score.time_signatures],
    )
    return melody_score, melody_keys


def is_strong_beat(time_tick: int, tpq: int) -> bool:
    return time_tick % tpq == 0


def is_half_beat(time_tick: int, tpq: int) -> bool:
    return time_tick % max(1, tpq // 2) == 0


def lerp(low: float, high: float, alpha: float) -> float:
    return low + (high - low) * alpha


def choose_control_token_name(bucket_idx: int) -> str:
    return f"CTRL_INTENSITY_{bucket_idx:02d}"


def bucket_for_intensity_value(intensity_value: float) -> tuple[int, float]:
    clipped = max(0.0, min(1.0, float(intensity_value)))
    bucket_idx = min(
        range(len(INTENSITY_LEVELS)),
        key=lambda idx: abs(INTENSITY_LEVELS[idx] - clipped),
    )
    return bucket_idx, float(INTENSITY_LEVELS[bucket_idx])


def desired_accompaniment_count(alpha: float, onset_tick: int, tpq: int) -> int:
    if is_strong_beat(onset_tick, tpq):
        if alpha < 0.16:
            return 1
        if alpha < 0.42:
            return 2
        if alpha < 0.74:
            return 3
        return 4
    if alpha < 0.16:
        return 0
    if alpha < 0.42:
        return 1
    if alpha < 0.74:
        return 2
    return 3


def select_accompaniment_notes(candidates: list, desired_count: int) -> list:
    if desired_count <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda note: (note.pitch, note.duration))
    picks = []
    indices = [0, len(ordered) - 1, len(ordered) // 2, 1, max(0, len(ordered) - 2)]
    seen = set()
    for idx in indices:
        if 0 <= idx < len(ordered) and idx not in seen:
            picks.append(ordered[idx])
            seen.add(idx)
        if len(picks) >= desired_count:
            break
    return picks[:desired_count]


def build_target(score: Score, melody_score: Score, melody_keys: set[tuple[int, int]], intensity_value: float) -> Score:
    alpha = max(0.0, min(1.0, float(intensity_value)))
    tpq = int(score.ticks_per_quarter)
    tempos = [Tempo(int(t.time), qpm=float(t.qpm)) for t in score.tempos]
    time_signatures = [TimeSignature(int(ts.time), int(ts.numerator), int(ts.denominator)) for ts in score.time_signatures]
    pedal_scale = lerp(0.6, 1.15, alpha)
    pedals = [
        Pedal(int(p.time), max(1, int(round(max(1, p.duration) * pedal_scale))))
        for p in score.tracks[0].pedals
    ]
    output_notes = []

    onset_lowest = {}
    for group in group_notes_by_onset(list(score.tracks[0].notes), tolerance_ticks=3):
        group = sorted(group, key=lambda note: note.pitch)
        onset = int(group[0].time)
        lowest = min(group, key=lambda note: note.pitch)
        onset_lowest[onset] = lowest

        melody_notes = []
        accompaniment = []
        for note in group:
            key = (int(note.time), int(note.pitch))
            if key in melody_keys:
                melody_notes.append(note)
            else:
                accompaniment.append(note)

        desired_count = desired_accompaniment_count(alpha, onset, tpq)
        long_candidates = [note for note in accompaniment if int(note.duration) >= tpq // 3]
        candidate_pool = long_candidates if long_candidates else accompaniment
        chosen_accompaniment = select_accompaniment_notes(candidate_pool, desired_count)

        for note in melody_notes:
            velocity = int(round(note.velocity * lerp(0.84, 1.15, alpha)))
            velocity = max(26, min(127, velocity))
            output_notes.append(
                Note(
                    int(note.time),
                    max(1, int(note.duration)),
                    int(note.pitch),
                    velocity,
                )
            )

        for note in chosen_accompaniment:
            min_duration = tpq // 6 if alpha < 0.35 else tpq // 10
            duration = max(1, int(note.duration))
            if duration < min_duration:
                continue
            velocity = int(round(note.velocity * lerp(0.48, 1.06, alpha)))
            velocity = max(18, min(122, velocity))
            output_notes.append(
                Note(
                    int(note.time),
                    duration,
                    int(note.pitch),
                    velocity,
                )
            )

    if alpha >= 0.55:
        melody_octave_threshold = lerp(0.55, 0.88, alpha)
        for note in melody_score.tracks[0].notes:
            onset_strength = is_strong_beat(int(note.time), tpq) or note.velocity >= 88
            if note.pitch <= 84 and onset_strength and alpha >= melody_octave_threshold:
                output_notes.append(
                    Note(
                        int(note.time),
                        max(1, int(note.duration)),
                        int(note.pitch) + 12,
                        max(38, min(118, int(round(note.velocity * lerp(0.68, 0.95, alpha))))),
                    )
                )

    if alpha >= 0.72:
        bass_threshold = lerp(0.72, 0.96, alpha)
        for onset, note in onset_lowest.items():
            if note.pitch >= 36 and (is_strong_beat(onset, tpq) or (alpha > 0.9 and is_half_beat(onset, tpq))):
                if alpha >= bass_threshold:
                    output_notes.append(
                        Note(
                            onset,
                            max(1, int(note.duration)),
                            int(note.pitch) - 12,
                            max(28, min(120, int(round(note.velocity * lerp(0.72, 0.98, alpha))))),
                        )
                    )

    dedup = {}
    for note in output_notes:
        key = (int(note.time), int(note.pitch))
        if key not in dedup or int(note.velocity) > int(dedup[key].velocity):
            dedup[key] = Note(int(note.time), max(1, int(note.duration)), int(note.pitch), int(note.velocity))

    final_notes = sorted(dedup.values(), key=lambda note: (note.time, note.pitch))
    return create_score_from_components(tpq, final_notes, pedals, tempos, time_signatures)


def evenly_spaced_starts(total_end: int, window_ticks: int, step_ticks: int, max_windows: int) -> list[int]:
    if total_end <= window_ticks:
        return [0]
    starts = list(range(0, max(1, total_end - window_ticks + 1), step_ticks))
    last_start = max(0, total_end - window_ticks)
    if starts[-1] != last_start:
        starts.append(last_start)
    if len(starts) <= max_windows:
        return starts
    idxs = np.linspace(0, len(starts) - 1, num=max_windows, dtype=int)
    unique = sorted({starts[idx] for idx in idxs})
    return unique


def build_sequence(
    source_ids: list[int],
    target_ids: list[int],
    intensity_bucket_idx: int,
    special_tokens: dict[str, int],
) -> tuple[list[int], int]:
    intensity_token = special_tokens[choose_control_token_name(intensity_bucket_idx)]
    seq = [
        intensity_token,
        special_tokens["SRC_START"],
        *source_ids,
        special_tokens["SRC_END"],
        special_tokens["TGT_START"],
        *target_ids,
        special_tokens["TGT_END"],
    ]
    loss_start = seq.index(special_tokens["TGT_START"]) + 1
    return seq, loss_start


def save_example(
    split: str,
    example_index: int,
    sequence: list[int],
    loss_start: int,
    metadata: dict,
):
    out_dir = EXAMPLES / split
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{example_index:07d}.npz"
    np.savez_compressed(
        path,
        ids=np.asarray(sequence, dtype=np.int32),
        loss_start=np.asarray(loss_start, dtype=np.int32),
    )
    metadata["path"] = str(path)
    return path


def clear_outputs():
    if EXAMPLES.exists():
        for path in EXAMPLES.rglob("*.npz"):
            path.unlink()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)


def main():
    global ARTIFACTS, EXAMPLES, TOKENIZER_PATH, META_PATH, MANIFEST_PATH, LOGS, STATUS_PATH
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    ARTIFACTS = run_root / "artifacts"
    EXAMPLES = ARTIFACTS / "examples"
    TOKENIZER_PATH = ARTIFACTS / "tokenizer.json"
    META_PATH = ARTIFACTS / "meta.json"
    MANIFEST_PATH = ARTIFACTS / "example_manifest.jsonl"
    LOGS = run_root / "logs"
    STATUS_PATH = LOGS / "dataset_status.json"

    source_root = args.source_root.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    clear_outputs()

    tokenizer = build_tokenizer()
    tokenizer.save(str(TOKENIZER_PATH))
    base_vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))
    special_tokens = make_special_tokens(base_vocab_size)
    vocab_size = base_vocab_size + len(special_tokens)

    source_files = sorted(list(source_root.glob("*.mid")) + list(source_root.glob("*.midi")))
    source_files = [path for path in source_files if path.is_file()]
    if args.max_files > 0:
        source_files = source_files[:args.max_files]
    if not source_files:
        raise SystemExit(f"no MIDI files found under {source_root}")

    split_by_group = split_groups(
        source_files,
        seed=args.seed,
        train_ratio=args.split_train,
        val_ratio=args.split_val,
        test_ratio=args.split_test,
    )

    write_status(
        {
            "phase": "startup",
            "source_root": str(source_root),
            "source_files": len(source_files),
            "vocab_size": vocab_size,
        }
    )
    print(
        f"[dataset] starting run_root={run_root} source_root={source_root} files={len(source_files)} "
        f"window_beats={args.window_beats} max_windows_per_file={args.max_windows_per_file}",
        flush=True,
    )

    manifest_rows = []
    counts = {"train": 0, "validation": 0, "test": 0}
    skipped = defaultdict(int)
    example_idx = defaultdict(int)
    build_start = time.time()

    for file_idx, midi_path in enumerate(source_files, start=1):
        split = split_by_group[base_piece_key(midi_path)]

        try:
            score = merge_score(Score(str(midi_path)))
        except Exception:
            skipped["load_error"] += 1
            continue

        if not score.tracks or not score.tracks[0].notes:
            skipped["empty_score"] += 1
            continue

        tpq = int(score.ticks_per_quarter)
        window_ticks = args.window_beats * tpq
        step_ticks = args.step_beats * tpq
        starts = evenly_spaced_starts(
            total_end=int(score.end()),
            window_ticks=window_ticks,
            step_ticks=step_ticks,
            max_windows=args.max_windows_per_file,
        )

        for window_idx, start_tick in enumerate(starts):
            end_tick = start_tick + window_ticks
            window = score_window(score, start_tick, end_tick)
            if not window.tracks or len(window.tracks[0].notes) < args.min_notes:
                skipped["too_few_notes"] += 1
                continue

            melody_score, melody_keys = extract_melody(window, onset_tolerance_ticks=max(1, tpq // 32))
            if len(melody_score.tracks[0].notes) < args.min_melody_notes:
                skipped["too_few_melody_notes"] += 1
                continue

            try:
                source_ids = extract_ids(tokenizer(melody_score))
            except Exception:
                skipped["source_tokenize_error"] += 1
                continue

            if len(source_ids) > args.max_source_tokens:
                skipped["source_too_long"] += 1
                continue

            for intensity_bucket_idx, requested_intensity in enumerate(INTENSITY_LEVELS):
                _, quantized_intensity = bucket_for_intensity_value(requested_intensity)
                target_score = build_target(
                    window,
                    melody_score,
                    melody_keys,
                    intensity_value=quantized_intensity,
                )
                try:
                    target_ids = extract_ids(tokenizer(target_score))
                except Exception:
                    skipped[f"bucket_{intensity_bucket_idx:02d}_target_tokenize_error"] += 1
                    continue

                if len(target_ids) > args.max_target_tokens:
                    skipped[f"bucket_{intensity_bucket_idx:02d}_target_too_long"] += 1
                    continue

                sequence, loss_start = build_sequence(
                    source_ids,
                    target_ids,
                    intensity_bucket_idx=intensity_bucket_idx,
                    special_tokens=special_tokens,
                )
                if len(sequence) > args.max_seq_len:
                    skipped["sequence_too_long"] += 1
                    continue

                metadata = {
                    "split": split,
                    "source_file": str(midi_path),
                    "group_key": base_piece_key(midi_path),
                    "window_index": window_idx,
                    "start_tick": int(start_tick),
                    "end_tick": int(end_tick),
                    "intensity_bucket_idx": int(intensity_bucket_idx),
                    "intensity_value": float(quantized_intensity),
                    "source_tokens": len(source_ids),
                    "target_tokens": len(target_ids),
                    "sequence_tokens": len(sequence),
                    "loss_start": int(loss_start),
                }

                save_example(split, example_idx[split], sequence, loss_start, metadata)
                example_idx[split] += 1
                counts[split] += 1
                manifest_rows.append(metadata)

        if file_idx % 128 == 0 or file_idx == len(source_files):
            elapsed = time.time() - build_start
            progress = file_idx / max(1, len(source_files))
            eta = (elapsed / max(progress, 1e-9)) * (1.0 - progress)
            payload = {
                "phase": "building",
                "processed_files": file_idx,
                "total_files": len(source_files),
                "counts": counts,
                "skipped": dict(skipped),
                "elapsed_sec": elapsed,
                "eta_sec": eta,
            }
            write_status(payload)
            print(
                f"[dataset] {file_idx}/{len(source_files)} ({progress*100:5.1f}%) "
                f"train={counts['train']} val={counts['validation']} test={counts['test']} "
                f"skipped={sum(skipped.values())} elapsed={fmt_seconds(elapsed)} "
                f"eta={fmt_seconds(eta)}",
                flush=True,
            )

    with MANIFEST_PATH.open("w") as fh:
        for row in manifest_rows:
            fh.write(json.dumps(row) + "\n")

    meta = {
        "dataset": "melody_intensity_editor",
        "run_root": str(run_root),
        "source_root": str(source_root),
        "source_file_count": len(source_files),
        "source_file_cap": args.max_files,
        "base_vocab_size": base_vocab_size,
        "vocab_size": vocab_size,
        "tokenizer_path": str(TOKENIZER_PATH),
        "special_tokens": special_tokens,
        "intensity_levels": INTENSITY_LEVELS,
        "max_seq_len": args.max_seq_len,
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
        "window_beats": args.window_beats,
        "step_beats": args.step_beats,
        "max_windows_per_file": args.max_windows_per_file,
        "counts": counts,
        "skipped": dict(skipped),
        "seed": args.seed,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    write_status({"phase": "finished", "meta": meta})

    total_elapsed = time.time() - build_start
    print(f"[dataset] build complete elapsed={fmt_seconds(total_elapsed)}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
