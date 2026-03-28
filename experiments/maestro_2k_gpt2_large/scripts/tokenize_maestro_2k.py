import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from miditok import REMI, TokenizerConfig

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = EXPERIMENT_ROOT / "data" / "splits"
ARTIFACTS = EXPERIMENT_ROOT / "artifacts"
TOKENS = ARTIFACTS / "tokens"
TOKENIZER_PATH = ARTIFACTS / "tokenizer.json"
META_PATH = ARTIFACTS / "meta.json"
STATUS_PATH = EXPERIMENT_ROOT / "logs" / "tokenize_status.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize the isolated Maestro 2k split into REMI token arrays."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Worker processes for tokenization.",
    )
    return parser.parse_args()


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


def write_status(payload: dict):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2))


def clear_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.npy"):
        old.unlink()


def tokenize_one(job):
    split, idx, src_str, out_dir_str = job
    src = Path(src_str)
    out_dir = Path(out_dir_str)

    try:
        tokenizer = build_tokenizer()
        tokens = tokenizer(str(src))

        if isinstance(tokens, list):
            if not tokens:
                return {"ok": False, "split": split, "file": src.name, "reason": "empty token list"}
            first = tokens[0]
            ids = np.asarray(first.ids if hasattr(first, "ids") else first, dtype=np.int32)
        else:
            ids = np.asarray(tokens.ids if hasattr(tokens, "ids") else tokens, dtype=np.int32)

        if len(ids) < 32:
            return {"ok": False, "split": split, "file": src.name, "reason": "too short"}

        out_path = out_dir / f"{idx:05d}_{src.stem}.npy"
        np.save(out_path, ids)

        return {
            "ok": True,
            "split": split,
            "file": src.name,
            "out": str(out_path),
            "num_tokens": int(len(ids)),
        }
    except Exception as exc:
        return {"ok": False, "split": split, "file": src.name, "reason": str(exc)}


def main():
    args = parse_args()
    tokenizer = build_tokenizer()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))

    jobs = []
    queued = {}
    for split in ["train", "validation", "test"]:
        split_dir = SPLITS / split
        out_dir = TOKENS / split
        clear_output_dir(out_dir)

        midi_files = sorted(list(split_dir.glob("*.mid")) + list(split_dir.glob("*.midi")))
        queued[split] = len(midi_files)
        for idx, src in enumerate(midi_files):
            jobs.append((split, idx, str(src.resolve()), str(out_dir.resolve())))

    if not jobs:
        raise SystemExit(f"no split MIDI files found under {SPLITS}")

    counts = {"train": 0, "validation": 0, "test": 0}
    skipped = 0
    processed = 0
    total = len(jobs)

    write_status(
        {
            "phase": "startup",
            "queued": queued,
            "processed": 0,
            "counts": counts,
            "skipped": skipped,
            "max_workers": args.max_workers,
        }
    )

    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(tokenize_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            processed += 1

            if result["ok"]:
                counts[result["split"]] += 1
            else:
                skipped += 1
                print(
                    f"skipped [{result['split']}] {result['file']}: {result['reason']}",
                    flush=True,
                )

            if processed % 100 == 0 or processed == total:
                payload = {
                    "phase": "tokenizing",
                    "queued": queued,
                    "processed": processed,
                    "total": total,
                    "counts": counts,
                    "skipped": skipped,
                }
                write_status(payload)
                print(
                    f"progress: {processed}/{total} | "
                    f"train={counts['train']} val={counts['validation']} test={counts['test']} "
                    f"skipped={skipped}",
                    flush=True,
                )

    meta = {
        "vocab_size": vocab_size,
        "counts": counts,
        "skipped": skipped,
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenization": "REMI",
        "dataset": "maestro_2k",
        "augmentation": False,
        "bpe": False,
        "parallel": True,
        "max_workers": args.max_workers,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    write_status({"phase": "finished", "meta": meta})

    print("tokenization complete", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
