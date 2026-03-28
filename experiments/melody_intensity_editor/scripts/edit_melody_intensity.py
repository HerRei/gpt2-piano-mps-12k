import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from miditok import REMI
from symusic import Note, Pedal, Score, Tempo, TimeSignature, Track
from transformers import GPT2LMHeadModel

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = EXPERIMENT_ROOT / "checkpoints" / "best"
DEFAULT_TOKENIZER = EXPERIMENT_ROOT / "artifacts" / "tokenizer.json"
DEFAULT_META = EXPERIMENT_ROOT / "artifacts" / "meta.json"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "exports"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edit a short melody phrase with one intensity value controlling loudness and intensity."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--melody-midi", type=Path, required=True)
    parser.add_argument(
        "--intensity-value",
        type=float,
        default=0.5,
        help="Requested intensity in the range 0.0 (soft/sparse) to 1.0 (loud/dense).",
    )
    parser.add_argument(
        "--intensity",
        type=str,
        default=None,
        help="Backward-compatible alias. Accepts calm, balanced, intense, or a numeric value.",
    )
    parser.add_argument("--extract-melody", action="store_true")
    parser.add_argument(
        "--window-beats",
        type=int,
        default=0,
        help="Optional phrase window size in beats. If omitted, long inputs are auto-fit using the dataset window size.",
    )
    parser.add_argument(
        "--window-position",
        choices=["start", "middle", "end"],
        default="end",
        help="Which part of a long melody to keep when auto-fitting to the token budget.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=680)
    parser.add_argument("--temperature", type=float, default=0.88)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("requested mps but MPS is not available")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokenizer(path: Path) -> REMI:
    if not path.exists():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    return REMI(params=path)


def extract_ids(seq) -> list[int]:
    if isinstance(seq, list):
        if len(seq) != 1:
            raise ValueError(f"expected single-track tokenization result, got {len(seq)} tracks")
        seq = seq[0]
    if not hasattr(seq, "ids"):
        raise ValueError("tokenization result is missing ids")
    return list(seq.ids)


def save_midi(tokenizer: REMI, token_ids: list[int], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    score = tokenizer.decode([token_ids])
    score.dump_midi(str(path.resolve()))


def clip_intensity(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_requested_intensity(raw_intensity: str | None, intensity_value: float) -> float:
    if raw_intensity is None:
        return clip_intensity(intensity_value)
    lowered = raw_intensity.strip().lower()
    legacy = {
        "calm": 0.0,
        "balanced": 0.5,
        "intense": 1.0,
    }
    if lowered in legacy:
        return legacy[lowered]
    try:
        return clip_intensity(float(raw_intensity))
    except ValueError as exc:
        raise ValueError(
            f"invalid intensity value '{raw_intensity}': expected calm, balanced, intense, or a number"
        ) from exc


def resolve_intensity_control(meta: dict, requested_intensity: float) -> tuple[int, float]:
    levels = meta.get("intensity_levels")
    if levels:
        clipped = clip_intensity(requested_intensity)
        bucket_idx = min(
            range(len(levels)),
            key=lambda idx: abs(float(levels[idx]) - clipped),
        )
        token_name = f"CTRL_INTENSITY_{bucket_idx:02d}"
        if token_name not in meta["special_tokens"]:
            raise KeyError(f"missing control token in meta: {token_name}")
        return int(meta["special_tokens"][token_name]), float(levels[bucket_idx])

    legacy_tokens = meta.get("special_tokens", {})
    if requested_intensity <= 0.25:
        return int(legacy_tokens["CTRL_CALM"]), 0.0
    if requested_intensity >= 0.75:
        return int(legacy_tokens["CTRL_INTENSE"]), 1.0
    return int(legacy_tokens["CTRL_BALANCED"]), 0.5


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


def trim_score_window(score: Score, window_beats: int, position: str) -> Score:
    if window_beats <= 0:
        return merge_score(score)
    tpq = int(score.ticks_per_quarter)
    window_ticks = max(1, int(window_beats) * tpq)
    total_end = int(score.end())
    if total_end <= window_ticks:
        return merge_score(score)
    if position == "start":
        start_tick = 0
    elif position == "middle":
        start_tick = max(0, (total_end - window_ticks) // 2)
    else:
        start_tick = max(0, total_end - window_ticks)
    window = score.trim(start_tick, start_tick + window_ticks)
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


def merge_score(score: Score) -> Score:
    notes = []
    pedals = []
    for track in score.tracks:
        notes.extend(
            Note(int(note.time), max(1, int(note.duration)), int(note.pitch), int(note.velocity))
            for note in track.notes
        )
        pedals.extend(
            Pedal(int(pedal.time), max(1, int(pedal.duration)))
            for pedal in track.pedals
        )
    notes.sort(key=lambda note: (note.time, note.pitch))
    pedals.sort(key=lambda pedal: (pedal.time, pedal.duration))
    tempos = [Tempo(int(t.time), qpm=float(t.qpm)) for t in score.tempos]
    time_signatures = [TimeSignature(int(ts.time), int(ts.numerator), int(ts.denominator)) for ts in score.time_signatures]
    return create_score_from_components(score.ticks_per_quarter, notes, pedals, tempos, time_signatures)


def extract_melody(score: Score) -> Score:
    merged = merge_score(score)
    tpq = int(merged.ticks_per_quarter)
    melody_notes = []
    last_pitch = None
    for group in group_notes_by_onset(list(merged.tracks[0].notes), tolerance_ticks=max(1, tpq // 32)):
        lead = max(group, key=lambda note: (note.pitch, note.velocity, note.duration))
        if last_pitch is not None and lead.pitch < last_pitch - 16:
            alternatives = [note for note in group if note.pitch >= last_pitch - 12]
            if alternatives:
                lead = max(alternatives, key=lambda note: (note.pitch, note.velocity, note.duration))
        melody_notes.append(Note(int(lead.time), max(1, int(lead.duration)), int(lead.pitch), int(lead.velocity)))
        last_pitch = lead.pitch
    tempos = [Tempo(int(t.time), qpm=float(t.qpm)) for t in merged.tempos]
    time_signatures = [TimeSignature(int(ts.time), int(ts.numerator), int(ts.denominator)) for ts in merged.time_signatures]
    return create_score_from_components(tpq, melody_notes, [], tempos, time_signatures)


def generate_target(
    model: GPT2LMHeadModel,
    tokenizer: REMI,
    meta: dict,
    device: torch.device,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> list[int]:
    tgt_end_id = int(meta["special_tokens"]["TGT_END"])
    blocked_specials = [
        token_id
        for name, token_id in meta["special_tokens"].items()
        if name != "TGT_END"
    ]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tgt_end_id,
            pad_token_id=int(meta["special_tokens"]["PAD"]),
            bad_words_ids=[[token_id] for token_id in blocked_specials],
            use_cache=True,
            renormalize_logits=True,
            return_dict_in_generate=False,
        )
    return generated[0].tolist()


def extract_target_ids(full_ids: list[int], meta: dict) -> list[int]:
    tgt_start = int(meta["special_tokens"]["TGT_START"])
    tgt_end = int(meta["special_tokens"]["TGT_END"])
    try:
        start = full_ids.index(tgt_start) + 1
    except ValueError as exc:
        raise ValueError("generated sequence is missing TGT_START") from exc
    if tgt_end in full_ids[start:]:
        end = full_ids.index(tgt_end, start)
    else:
        end = len(full_ids)
    return [token_id for token_id in full_ids[start:end] if token_id < int(meta["base_vocab_size"])]


def fit_source_phrase(
    tokenizer: REMI,
    source_score: Score,
    max_source_tokens: int,
    default_window_beats: int,
    position: str,
) -> tuple[Score, list[int], int]:
    source_score = merge_score(source_score)
    full_ids = extract_ids(tokenizer(source_score))
    if len(full_ids) <= max_source_tokens:
        return source_score, full_ids, 0

    attempts = []
    if default_window_beats > 0:
        attempts.append(default_window_beats)
    attempts.extend([12, 8, 6, 4, 2])

    seen = set()
    for window_beats in attempts:
        if window_beats in seen:
            continue
        seen.add(window_beats)
        candidate_score = trim_score_window(source_score, window_beats=window_beats, position=position)
        candidate_ids = extract_ids(tokenizer(candidate_score))
        if len(candidate_ids) <= max_source_tokens:
            return candidate_score, candidate_ids, window_beats

    raise ValueError(
        f"source melody tokenized to {len(full_ids)} tokens, which exceeds max_source_tokens={max_source_tokens} "
        "even after phrase auto-fitting"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))

    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    meta_path = args.meta.expanduser().resolve()
    melody_midi = args.melody_midi.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not melody_midi.exists():
        raise FileNotFoundError(f"melody midi not found: {melody_midi}")

    tokenizer = load_tokenizer(tokenizer_path)
    meta = json.loads(meta_path.read_text())
    requested_intensity = parse_requested_intensity(args.intensity, args.intensity_value)

    source_score = Score(str(melody_midi))
    if args.extract_melody:
        source_score = extract_melody(source_score)
    else:
        source_score = merge_score(source_score)
    source_score, source_ids, applied_window_beats = fit_source_phrase(
        tokenizer=tokenizer,
        source_score=source_score,
        max_source_tokens=int(meta["max_source_tokens"]),
        default_window_beats=int(args.window_beats or meta.get("window_beats", 16)),
        position=args.window_position,
    )

    intensity_token, quantized_intensity = resolve_intensity_control(meta, requested_intensity)

    prompt_ids = [
        intensity_token,
        int(meta["special_tokens"]["SRC_START"]),
        *source_ids,
        int(meta["special_tokens"]["SRC_END"]),
        int(meta["special_tokens"]["TGT_START"]),
    ]

    device = choose_device(args.device)
    model = GPT2LMHeadModel.from_pretrained(checkpoint, local_files_only=True)
    model.to(device)
    model.eval()
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.generation_config.pad_token_id = int(meta["special_tokens"]["PAD"])

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

    name = args.name or f"edit_{quantized_intensity:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / f"{name}_melody.mid"
    output_path = output_dir / f"{name}.mid"
    token_path = output_dir / f"{name}.npy"
    meta_out_path = output_dir / f"{name}.json"

    prompt_score = source_score
    prompt_score.dump_midi(str(prompt_path.resolve()))
    save_midi(tokenizer, target_ids, output_path)
    np.save(token_path, np.asarray(target_ids, dtype=np.int32))

    metadata = {
        "checkpoint": str(checkpoint),
        "melody_midi": str(melody_midi),
        "requested_intensity_value": requested_intensity,
        "quantized_intensity_value": quantized_intensity,
        "device": str(device),
        "source_tokens": len(source_ids),
        "applied_window_beats": applied_window_beats,
        "window_position": args.window_position,
        "generated_target_tokens": len(target_ids),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "extract_melody": bool(args.extract_melody),
        "prompt_midi": str(prompt_path),
        "output_midi": str(output_path),
        "token_ids_npy": str(token_path),
    }
    meta_out_path.write_text(json.dumps(metadata, indent=2))

    print(f"prompt_midi: {prompt_path}")
    print(f"output_midi: {output_path}")
    print(f"metadata: {meta_out_path}")


if __name__ == "__main__":
    main()
