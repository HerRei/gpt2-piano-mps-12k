import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from symusic import Note, Pedal, Score, Tempo, TimeSignature
from transformers import GPT2LMHeadModel

from edit_melody_intensity import (
    convert_score_tpq,
    DEFAULT_CHECKPOINT,
    DEFAULT_META,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TOKENIZER,
    choose_device,
    create_score_from_components,
    extract_melody,
    extract_target_ids,
    fit_source_phrase,
    generate_target,
    load_tokenizer,
    merge_score,
    normalize_score_start,
    resolve_intensity_control,
    set_seed,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prototype phrase-by-phrase live test pipeline for the melody intensity editor. "
            "It windows a MIDI file, applies an intensity schedule over time, measures latency, "
            "and writes stitched outputs plus a timing report."
        )
    )
    parser.add_argument("--input-midi", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--extract-melody", action="store_true")
    parser.add_argument(
        "--window-beats",
        type=int,
        default=0,
        help="Phrase size in beats. Defaults to the dataset window size from meta.json.",
    )
    parser.add_argument(
        "--hop-beats",
        type=int,
        default=0,
        help="Step size between windows in beats. Defaults to window-beats. Overlap is not supported yet.",
    )
    parser.add_argument(
        "--intensity-schedule",
        type=str,
        default="0:0.5",
        help="Comma-separated beat:value points, for example 0:0.2,16:0.85,32:0.35",
    )
    parser.add_argument(
        "--schedule-mode",
        choices=["step", "linear"],
        default="step",
        help="How to interpret the intensity schedule between points.",
    )
    parser.add_argument(
        "--window-position",
        choices=["start", "middle", "end"],
        default="end",
        help="Fallback phrase fitting strategy if a window exceeds the source token budget.",
    )
    parser.add_argument("--max-windows", type=int, default=0, help="Optional cap for a quick smoke test.")
    parser.add_argument("--max-new-tokens", type=int, default=680)
    parser.add_argument("--temperature", type=float, default=0.88)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Skip model inference and write only the timing plan.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "live_test")
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def clip_intensity(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_intensity_schedule(raw: str) -> list[tuple[float, float]]:
    points = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(
                f"invalid schedule point '{piece}': expected beat:value, for example 16:0.85"
            )
        beat_raw, value_raw = piece.split(":", 1)
        beat = float(beat_raw.strip())
        value = clip_intensity(float(value_raw.strip()))
        points.append((beat, value))
    if not points:
        raise ValueError("intensity schedule is empty")
    points.sort(key=lambda item: item[0])
    return points


def intensity_for_beat(points: list[tuple[float, float]], beat: float, mode: str) -> float:
    if beat <= points[0][0]:
        return points[0][1]
    if beat >= points[-1][0]:
        return points[-1][1]

    for idx in range(1, len(points)):
        left_beat, left_value = points[idx - 1]
        right_beat, right_value = points[idx]
        if beat <= right_beat:
            if mode == "step":
                if beat == right_beat:
                    return right_value
                return left_value
            if right_beat <= left_beat:
                return left_value
            ratio = (beat - left_beat) / (right_beat - left_beat)
            return left_value + (right_value - left_value) * ratio
    return points[-1][1]


def trim_window(score: Score, start_tick: int, end_tick: int) -> Score:
    window = score.trim(start_tick, end_tick)
    window.shift_time(-start_tick)
    return merge_score(window)


def make_window_ranges(score: Score, window_beats: int, hop_beats: int) -> list[tuple[int, int]]:
    if window_beats <= 0:
        raise ValueError("window-beats must be positive")
    if hop_beats <= 0:
        raise ValueError("hop-beats must be positive")
    if hop_beats < window_beats:
        raise ValueError("overlapping hops are not supported in this prototype yet; use hop-beats >= window-beats")

    tpq = int(score.ticks_per_quarter)
    window_ticks = window_beats * tpq
    hop_ticks = hop_beats * tpq
    total_end = int(score.end())
    if total_end <= 0:
        return []

    windows = []
    start_tick = 0
    while start_tick < total_end:
        end_tick = min(total_end, start_tick + window_ticks)
        windows.append((start_tick, end_tick))
        start_tick += hop_ticks
    return windows


def copy_tempos(score: Score) -> list[Tempo]:
    tempos = [Tempo(int(t.time), qpm=float(t.qpm)) for t in score.tempos]
    return tempos or [Tempo(0, qpm=120.0)]


def copy_time_signatures(score: Score) -> list[TimeSignature]:
    signatures = [
        TimeSignature(int(ts.time), int(ts.numerator), int(ts.denominator))
        for ts in score.time_signatures
    ]
    return signatures or [TimeSignature(0, 4, 4)]


def ticks_to_seconds(score: Score, start_tick: int, end_tick: int) -> float:
    if end_tick <= start_tick:
        return 0.0

    tpq = int(score.ticks_per_quarter)
    tempos = sorted(copy_tempos(score), key=lambda item: int(item.time))
    if not tempos or int(tempos[0].time) > 0:
        tempos.insert(0, Tempo(0, qpm=120.0))

    total_seconds = 0.0
    for idx, tempo in enumerate(tempos):
        seg_start = max(start_tick, int(tempo.time))
        seg_end = end_tick
        if idx + 1 < len(tempos):
            seg_end = min(seg_end, int(tempos[idx + 1].time))
        if seg_end <= seg_start:
            continue
        total_seconds += (seg_end - seg_start) * (60.0 / (float(tempo.qpm) * tpq))
    return total_seconds


def shifted_notes_and_pedals(score: Score, shift_ticks: int) -> tuple[list[Note], list[Pedal]]:
    notes = []
    pedals = []
    for track in score.tracks:
        notes.extend(
            Note(
                int(note.time) + shift_ticks,
                max(1, int(note.duration)),
                int(note.pitch),
                int(note.velocity),
            )
            for note in track.notes
        )
        pedals.extend(
            Pedal(
                int(pedal.time) + shift_ticks,
                max(1, int(pedal.duration)),
            )
            for pedal in track.pedals
        )
    return notes, pedals


def decode_target_score(tokenizer, target_ids: list[int]) -> Score:
    return normalize_score_start(tokenizer.decode([target_ids]))


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))

    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    meta_path = args.meta.expanduser().resolve()
    input_midi = args.input_midi.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not input_midi.exists():
        raise FileNotFoundError(f"input midi not found: {input_midi}")

    tokenizer = load_tokenizer(tokenizer_path)
    meta = json.loads(meta_path.read_text())
    schedule = parse_intensity_schedule(args.intensity_schedule)

    source_score = Score(str(input_midi))
    if args.extract_melody:
        source_score = extract_melody(source_score)
    else:
        source_score = normalize_score_start(source_score)

    window_beats = int(args.window_beats or meta.get("window_beats", 16))
    hop_beats = int(args.hop_beats or window_beats)
    windows = make_window_ranges(source_score, window_beats=window_beats, hop_beats=hop_beats)
    if args.max_windows > 0:
        windows = windows[:args.max_windows]
    if not windows:
        raise RuntimeError("input midi produced no usable windows")

    run_name = args.name or f"live_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    windows_dir = run_dir / "windows"
    run_dir.mkdir(parents=True, exist_ok=True)
    windows_dir.mkdir(parents=True, exist_ok=True)

    source_path = run_dir / f"{run_name}_source.mid"
    source_score.dump_midi(str(source_path.resolve()))

    device = choose_device(args.device)
    model = None
    if not args.dry_run:
        model = GPT2LMHeadModel.from_pretrained(checkpoint, local_files_only=True)
        model.to(device)
        model.eval()
        model.config.use_cache = True
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = int(meta["special_tokens"]["PAD"])

    print(f"[live-test] input_midi={input_midi}", flush=True)
    print(f"[live-test] windows={len(windows)} window_beats={window_beats} hop_beats={hop_beats}", flush=True)
    print(f"[live-test] output_dir={run_dir}", flush=True)
    if args.dry_run:
        print("[live-test] dry-run enabled: skipping model inference", flush=True)
    else:
        print(f"[live-test] device={device}", flush=True)

    stitched_notes = []
    stitched_pedals = []
    window_rows = []

    for window_index, (start_tick, end_tick) in enumerate(windows):
        start_beat = start_tick / max(1, int(source_score.ticks_per_quarter))
        intensity_value = intensity_for_beat(schedule, beat=start_beat, mode=args.schedule_mode)
        budget_sec = ticks_to_seconds(source_score, start_tick, end_tick)
        window_score = trim_window(source_score, start_tick=start_tick, end_tick=end_tick)
        window_path = windows_dir / f"window_{window_index:03d}_source.mid"
        window_score.dump_midi(str(window_path.resolve()))

        target_ids = []
        target_path = None
        metadata_path = windows_dir / f"window_{window_index:03d}.json"
        prompt_applied_window_beats = 0
        quantized_intensity = None
        source_tokens = None
        generated_target_tokens = None
        error_message = None

        start_time = time.perf_counter()
        try:
            prompt_score, source_ids, prompt_applied_window_beats = fit_source_phrase(
                tokenizer=tokenizer,
                source_score=window_score,
                max_source_tokens=int(meta["max_source_tokens"]),
                default_window_beats=window_beats,
                position=args.window_position,
            )
            source_tokens = len(source_ids)

            if not args.dry_run:
                intensity_token, quantized_intensity = resolve_intensity_control(meta, intensity_value)
                prompt_ids = [
                    intensity_token,
                    int(meta["special_tokens"]["SRC_START"]),
                    *source_ids,
                    int(meta["special_tokens"]["SRC_END"]),
                    int(meta["special_tokens"]["TGT_START"]),
                ]
                full_ids = generate_target(
                    model=model,
                    tokenizer=tokenizer,
                    meta=meta,
                    device=device,
                    prompt_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                target_ids = extract_target_ids(full_ids, meta)
                if not target_ids:
                    raise RuntimeError("generated output is empty")

                generated_target_tokens = len(target_ids)
                quantized_intensity = float(quantized_intensity)
                target_score = decode_target_score(tokenizer, target_ids)
                target_score = convert_score_tpq(target_score, int(source_score.ticks_per_quarter))
                target_path = windows_dir / f"window_{window_index:03d}_output.mid"
                target_score.dump_midi(str(target_path.resolve()))
                notes, pedals = shifted_notes_and_pedals(target_score, shift_ticks=start_tick)
                stitched_notes.extend(notes)
                stitched_pedals.extend(pedals)
        except Exception as exc:
            error_message = str(exc)

        latency_sec = time.perf_counter() - start_time
        within_budget = args.dry_run or latency_sec <= budget_sec

        window_row = {
            "window_index": window_index,
            "start_tick": start_tick,
            "end_tick": end_tick,
            "start_beat": start_beat,
            "end_beat": end_tick / max(1, int(source_score.ticks_per_quarter)),
            "requested_intensity_value": intensity_value,
            "quantized_intensity_value": quantized_intensity,
            "budget_sec": budget_sec,
            "latency_sec": latency_sec,
            "within_budget": within_budget,
            "applied_window_beats": prompt_applied_window_beats,
            "source_tokens": source_tokens,
            "generated_target_tokens": generated_target_tokens,
            "error": error_message,
            "source_window_midi": str(window_path),
            "output_window_midi": str(target_path) if target_path is not None else None,
        }
        metadata_path.write_text(json.dumps(window_row, indent=2))
        window_row["metadata_path"] = str(metadata_path)
        window_rows.append(window_row)

        status = "ok" if error_message is None else f"error={error_message}"
        print(
            f"[live-test] window={window_index + 1}/{len(windows)} "
            f"beat={start_beat:.1f} intensity={intensity_value:.2f} "
            f"latency={latency_sec:.3f}s budget={budget_sec:.3f}s within_budget={within_budget} {status}",
            flush=True,
        )

    stitched_output_path = None
    if stitched_notes:
        stitched_notes.sort(key=lambda note: (note.time, note.pitch, note.duration, note.velocity))
        stitched_pedals.sort(key=lambda pedal: (pedal.time, pedal.duration))
        stitched_score = create_score_from_components(
            int(source_score.ticks_per_quarter),
            stitched_notes,
            stitched_pedals,
            copy_tempos(source_score),
            copy_time_signatures(source_score),
        )
        stitched_output_path = run_dir / f"{run_name}_stitched.mid"
        stitched_score.dump_midi(str(stitched_output_path.resolve()))

    successful = [row for row in window_rows if row["error"] is None]
    avg_latency = sum(row["latency_sec"] for row in successful) / max(1, len(successful))
    max_latency = max((row["latency_sec"] for row in successful), default=0.0)
    within_budget_count = sum(1 for row in successful if row["within_budget"])

    summary = {
        "input_midi": str(input_midi),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "dry_run": bool(args.dry_run),
        "extract_melody": bool(args.extract_melody),
        "window_beats": window_beats,
        "hop_beats": hop_beats,
        "intensity_schedule": schedule,
        "schedule_mode": args.schedule_mode,
        "windows_total": len(window_rows),
        "windows_successful": len(successful),
        "windows_within_budget": within_budget_count,
        "avg_latency_sec": avg_latency,
        "max_latency_sec": max_latency,
        "source_midi": str(source_path),
        "stitched_output_midi": str(stitched_output_path) if stitched_output_path is not None else None,
        "windows": window_rows,
    }
    summary_path = run_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[live-test] summary={summary_path}", flush=True)
    if stitched_output_path is not None:
        print(f"[live-test] stitched_output={stitched_output_path}", flush=True)


if __name__ == "__main__":
    main()
