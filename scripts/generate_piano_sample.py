import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from miditok import REMI
from symusic import Score
from transformers import GPT2LMHeadModel

from _paths import ROOT
from generation_utils import (
    clamp01,
    longest_run,
    parse_int_list,
    parse_str_list,
    repeated_ngram_ratio,
    slice_prompt,
    target_score,
)


# Keep generation off MPS so it does not contend with training.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

TOKENS = ROOT / "artifacts" / "tokens"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best"
DEFAULT_TOKENIZER = ROOT / "artifacts" / "tokenizer.json"
DEFAULT_OUTPUT_DIR = ROOT / "exports" / "generated"
PRESETS = {
    "polished": {
        "temperature": 0.88,
        "top_k": 40,
        "top_p": 0.92,
        "repetition_penalty": 1.12,
    },
    "balanced": {
        "temperature": 0.92,
        "top_k": 56,
        "top_p": 0.94,
        "repetition_penalty": 1.08,
    },
    "exciting": {
        "temperature": 1.0,
        "top_k": 80,
        "top_p": 0.97,
        "repetition_penalty": 1.08,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CPU-only piano MIDI sample from a checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint directory produced by save_pretrained().",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="Path to the Miditok tokenizer json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where MIDI and metadata files will be written.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional output stem. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--prompt-midi",
        type=Path,
        default=None,
        help="Optional MIDI file to use as the prompt.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=Path,
        default=None,
        help="Optional .npy token file to use as the prompt.",
    )
    parser.add_argument(
        "--prompt-split",
        choices=["train", "validation", "test"],
        default="validation",
        help="Prompt source split when no explicit prompt path is provided.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="Index into the prompt split when no explicit prompt path is provided.",
    )
    parser.add_argument(
        "--search-prompt-indices",
        type=str,
        default=None,
        help="Optional comma-separated prompt indices to try and rank.",
    )
    parser.add_argument(
        "--prompt-position",
        choices=["start", "middle", "random"],
        default="start",
        help="Where to slice the prompt from within the source token file.",
    )
    parser.add_argument(
        "--search-prompt-positions",
        type=str,
        default=None,
        help="Optional comma-separated prompt positions to try when searching.",
    )
    parser.add_argument(
        "--prompt-offset",
        type=int,
        default=None,
        help="Optional exact token offset for the prompt slice.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=256,
        help="Number of prompt tokens to keep before generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Number of new tokens to sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.95,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="Sampling preset tuned for a particular style. Overrides the sampling args above.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff. Set to 0 to disable.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling cutoff.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Penalty applied to already generated tokens.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Torch CPU thread count. Keep low to avoid disturbing training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt selection and sampling.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Number of independent samples to generate from the same prompt.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokenizer(path: Path) -> REMI:
    if not path.exists():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    return REMI(params=path)


def apply_sampling_preset(args):
    if args.preset is None:
        return
    preset = PRESETS[args.preset]
    args.temperature = preset["temperature"]
    args.top_k = preset["top_k"]
    args.top_p = preset["top_p"]
    args.repetition_penalty = preset["repetition_penalty"]


def choose_prompt_file(split: str, index: int) -> Path:
    files = sorted((TOKENS / split).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no token files found under {(TOKENS / split)}")
    if index < 0 or index >= len(files):
        raise IndexError(f"prompt index {index} is out of range for split {split} ({len(files)} files)")
    return files[index]


def extract_ids_from_tokseq(seq) -> list[int]:
    if isinstance(seq, list):
        if len(seq) != 1:
            raise ValueError(f"expected a single-track prompt, got {len(seq)} tracks")
        seq = seq[0]
    if not hasattr(seq, "ids"):
        raise TypeError(f"unsupported token sequence type: {type(seq)!r}")
    return list(seq.ids)


def load_prompt_source_ids(args, tokenizer: REMI, prompt_index: Optional[int] = None) -> tuple[list[int], str]:
    if args.prompt_midi is not None and args.prompt_tokens is not None:
        raise ValueError("use only one of --prompt-midi or --prompt-tokens")

    if args.prompt_midi is not None:
        score = Score(args.prompt_midi)
        ids = extract_ids_from_tokseq(tokenizer(score))
        return ids, str(args.prompt_midi)

    if args.prompt_tokens is not None:
        ids = np.load(args.prompt_tokens).astype(np.int64).tolist()
        return ids, str(args.prompt_tokens)

    resolved_prompt_index = args.prompt_index if prompt_index is None else prompt_index
    prompt_path = choose_prompt_file(args.prompt_split, resolved_prompt_index)
    ids = np.load(prompt_path).astype(np.int64).tolist()
    return ids, str(prompt_path)


def save_midi(tokenizer: REMI, token_ids: list[int], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    score = tokenizer.decode([token_ids])
    score.dump_midi(str(path.resolve()))


def analyze_candidate(tokenizer: REMI, full_ids: list[int], prompt_len: int) -> dict:
    generated_ids = full_ids[prompt_len:]
    analysis = {
        "generated_unique_token_ratio": 0.0,
        "generated_longest_token_run": 0,
        "generated_repeated_4gram_ratio": 0.0,
        "note_count": 0,
        "notes_per_beat": 0.0,
        "pitch_span": 0,
        "unique_pitches": 0,
        "velocity_std": 0.0,
        "tempo_changes": 0,
        "time_signature_changes": 0,
        "heuristic_score": -1e9,
        "analysis_error": None,
    }

    if generated_ids:
        analysis["generated_unique_token_ratio"] = len(set(generated_ids)) / len(generated_ids)
        analysis["generated_longest_token_run"] = longest_run(generated_ids)
        analysis["generated_repeated_4gram_ratio"] = repeated_ngram_ratio(generated_ids, 4)

    try:
        score = tokenizer.decode([full_ids])
        if not score.tracks:
            analysis["analysis_error"] = "decoded score has no tracks"
            return analysis

        track = max(score.tracks, key=lambda tr: len(tr.notes))
        notes = sorted(track.notes, key=lambda n: (n.time, n.pitch))
        if not notes:
            analysis["analysis_error"] = "decoded score has no notes"
            return analysis

        start_time = min(note.time for note in notes)
        end_time = max(note.time + note.duration for note in notes)
        beats = max((end_time - start_time) / max(score.tpq, 1), 1e-6)
        velocities = np.asarray([note.velocity for note in notes], dtype=np.float32)
        pitches = [note.pitch for note in notes]
        note_starts = [note.time for note in notes]
        positive_iois = [b - a for a, b in zip(note_starts, note_starts[1:]) if b - a > 0]

        analysis["note_count"] = len(notes)
        analysis["notes_per_beat"] = len(notes) / beats
        analysis["pitch_span"] = max(pitches) - min(pitches)
        analysis["unique_pitches"] = len(set(pitches))
        analysis["velocity_std"] = float(np.std(velocities))
        analysis["tempo_changes"] = len(score.tempos)
        analysis["time_signature_changes"] = len(score.time_signatures)
        analysis["unique_iois"] = len(set(positive_iois))

        quality = 0.0
        quality += 2.2 * target_score(analysis["notes_per_beat"], target=5.0, tolerance=3.0)
        quality += 1.6 * target_score(analysis["pitch_span"], target=50.0, tolerance=24.0)
        quality += 1.2 * target_score(analysis["unique_pitches"], target=30.0, tolerance=18.0)
        quality += 0.9 * target_score(analysis["velocity_std"], target=14.0, tolerance=10.0)
        quality += 0.8 * target_score(analysis["generated_unique_token_ratio"], target=0.38, tolerance=0.22)
        quality += 0.8 * target_score(float(analysis["unique_iois"]), target=12.0, tolerance=10.0)
        quality -= 2.5 * analysis["generated_repeated_4gram_ratio"]
        quality -= 0.12 * max(0, analysis["generated_longest_token_run"] - 4)
        quality -= 0.25 * max(0, analysis["tempo_changes"] - 2)
        quality -= 0.2 * max(0, analysis["time_signature_changes"] - 2)

        analysis["heuristic_score"] = float(round(quality, 6))
        return analysis

    except Exception as exc:
        analysis["analysis_error"] = repr(exc)
        return analysis


def generate_candidate(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    tokenizer: REMI,
    args,
    candidate_seed: int,
) -> list[int]:
    set_seed(candidate_seed)
    special_token_ids = getattr(tokenizer, "special_tokens_ids", []) or []
    bad_words_ids = [[tok_id] for tok_id in special_token_ids if tok_id is not None]

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=None if args.top_k <= 0 else args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_cache=True,
            bad_words_ids=bad_words_ids or None,
            renormalize_logits=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=False,
        )

    return generated[0].tolist()


def main():
    args = parse_args()
    apply_sampling_preset(args)

    set_seed(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))

    checkpoint = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    tokenizer = load_tokenizer(tokenizer_path)

    model = GPT2LMHeadModel.from_pretrained(checkpoint, local_files_only=True)
    model.to("cpu")
    model.eval()
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    context_limit = int(getattr(model.config, "n_positions", getattr(model.config, "n_ctx", 2048)))

    name = args.name or f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    search_prompt_indices = parse_int_list(args.search_prompt_indices)
    if not search_prompt_indices:
        search_prompt_indices = [args.prompt_index]

    search_positions = parse_str_list(args.search_prompt_positions)
    if not search_positions:
        search_positions = [args.prompt_position]

    prompt_specs = []
    if args.prompt_midi is not None or args.prompt_tokens is not None:
        prompt_specs.append({
            "prompt_index": None,
            "prompt_position": args.prompt_position,
            "prompt_offset": args.prompt_offset,
        })
    else:
        for prompt_index in search_prompt_indices:
            for prompt_position in search_positions:
                prompt_specs.append({
                    "prompt_index": prompt_index,
                    "prompt_position": prompt_position,
                    "prompt_offset": args.prompt_offset,
                })

    print(f"device: cpu")
    print(f"checkpoint: {checkpoint}")
    print(f"preset: {args.preset or 'custom'}")
    print(f"num_candidates_per_prompt: {args.num_candidates}")
    print(f"prompt_specs: {len(prompt_specs)}")

    all_metadata = []
    overall_candidate = 0
    for prompt_spec_idx, prompt_spec in enumerate(prompt_specs, start=1):
        prompt_seed = args.seed + prompt_spec_idx * 1000
        set_seed(prompt_seed)

        full_prompt_ids, prompt_source = load_prompt_source_ids(
            args,
            tokenizer,
            prompt_index=prompt_spec["prompt_index"],
        )
        prompt_ids, prompt_slice_meta = slice_prompt(
            full_prompt_ids=full_prompt_ids,
            prompt_length=args.prompt_length,
            max_context=context_limit,
            max_new_tokens=args.max_new_tokens,
            prompt_position=prompt_spec["prompt_position"],
            prompt_offset=prompt_spec["prompt_offset"],
        )
        prompt_len = len(prompt_ids)

        prompt_label_bits = []
        if prompt_spec["prompt_index"] is not None:
            prompt_label_bits.append(f"p{prompt_spec['prompt_index']:02d}")
        prompt_label_bits.append(prompt_spec["prompt_position"])
        prompt_label = "_".join(prompt_label_bits)

        prompt_midi_path = output_dir / f"{name}_{prompt_label}_prompt.mid"
        save_midi(tokenizer, prompt_ids, prompt_midi_path)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cpu")

        print(f"prompt_{prompt_spec_idx}_source: {prompt_source}")
        print(f"prompt_{prompt_spec_idx}_tokens: {prompt_len}")
        print(f"prompt_{prompt_spec_idx}_position: {prompt_spec['prompt_position']}")
        print(f"prompt_{prompt_spec_idx}_midi: {prompt_midi_path}")

        for candidate_idx in range(args.num_candidates):
            overall_candidate += 1
            candidate_seed = args.seed + overall_candidate
            full_ids = generate_candidate(
                model=model,
                input_ids=input_ids,
                tokenizer=tokenizer,
                args=args,
                candidate_seed=candidate_seed,
            )
            new_token_count = len(full_ids) - prompt_len

            if len(prompt_specs) == 1 and args.num_candidates == 1:
                candidate_stem = output_dir / name
            else:
                candidate_stem = output_dir / f"{name}_{prompt_label}_c{candidate_idx + 1:02d}"

            sample_midi_path = candidate_stem.with_suffix(".mid")
            tokens_path = candidate_stem.with_suffix(".npy")
            meta_path = candidate_stem.with_suffix(".json")

            save_midi(tokenizer, full_ids, sample_midi_path)
            np.save(tokens_path, np.asarray(full_ids, dtype=np.int32))

            analysis = analyze_candidate(tokenizer, full_ids, prompt_len)
            metadata = {
                "checkpoint": str(checkpoint),
                "tokenizer": str(tokenizer_path),
                "device": "cpu",
                "cpu_threads": int(args.cpu_threads),
                "seed": int(candidate_seed),
                "prompt_index": prompt_spec["prompt_index"],
                "candidate_index": candidate_idx + 1,
                "prompt_source": prompt_source,
                "prompt_tokens": prompt_len,
                "generated_tokens": int(new_token_count),
                "total_tokens": int(len(full_ids)),
                "context_limit": context_limit,
                "preset": args.preset,
                "temperature": float(args.temperature),
                "top_k": None if args.top_k <= 0 else int(args.top_k),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
                "prompt_midi": str(prompt_midi_path),
                "sample_midi": str(sample_midi_path),
                "token_ids_npy": str(tokens_path),
                **prompt_slice_meta,
                **analysis,
            }
            meta_path.write_text(json.dumps(metadata, indent=2))
            all_metadata.append(metadata)

            print(
                f"candidate_{overall_candidate}_score: "
                f"{metadata['heuristic_score']:.4f}"
            )
            print(f"candidate_{overall_candidate}_sample_midi: {sample_midi_path}")
            print(f"candidate_{overall_candidate}_metadata_json: {meta_path}")

    all_metadata.sort(key=lambda item: item["heuristic_score"], reverse=True)
    for rank, metadata in enumerate(all_metadata, start=1):
        metadata["rank"] = rank

    summary_path = output_dir / f"{name}_summary.json"
    summary = {
        "checkpoint": str(checkpoint),
        "num_prompt_specs": len(prompt_specs),
        "num_candidates_total": len(all_metadata),
        "preset": args.preset,
        "candidates": all_metadata,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary_json: {summary_path}")

    if all_metadata:
        best = all_metadata[0]
        print(f"best_score: {best['heuristic_score']:.4f}")
        print(f"best_sample_midi: {best['sample_midi']}")


if __name__ == "__main__":
    main()
