from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
from miditok import REMI, TokenizerConfig

from _paths import ROOT

TRAIN_SRC = ROOT / "data" / "augmented" / "train"
VAL_SRC = ROOT / "data" / "splits" / "validation"
TEST_SRC = ROOT / "data" / "splits" / "test"

OUT = ROOT / "artifacts" / "tokens"
TOK_PATH = ROOT / "artifacts" / "tokenizer.json"
META_PATH = ROOT / "artifacts" / "meta.json"
STATUS_PATH = ROOT / "logs" / "tokenize_status.json"

MAX_WORKERS = 8

for split in ["train", "validation", "test"]:
    (OUT / split).mkdir(parents=True, exist_ok=True)


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


def tokenize_one(job):
    split, idx, src_str = job
    src = Path(src_str)
    out_dir = OUT / split

    try:
        tokenizer = build_tokenizer()
        tokens = tokenizer(str(src))

        if isinstance(tokens, list):
            if len(tokens) == 0:
                return {"ok": False, "split": split, "file": src.name, "reason": "empty token list"}
            first = tokens[0]
            ids = np.asarray(first.ids if hasattr(first, "ids") else first, dtype=np.int32)
        else:
            ids = np.asarray(tokens.ids if hasattr(tokens, "ids") else tokens, dtype=np.int32)

        if len(ids) < 32:
            return {"ok": False, "split": split, "file": src.name, "reason": "too short"}

        out_name = f"{idx:05d}_{src.stem}.npy"
        out_path = out_dir / out_name
        np.save(out_path, ids)

        return {
            "ok": True,
            "split": split,
            "file": src.name,
            "out": str(out_path),
            "num_tokens": int(len(ids)),
        }

    except Exception as e:
        return {"ok": False, "split": split, "file": src.name, "reason": str(e)}


def main():
    tokenizer = build_tokenizer()
    tokenizer.save(str(TOK_PATH))
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))

    split_sources = {
        "train": TRAIN_SRC,
        "validation": VAL_SRC,
        "test": TEST_SRC,
    }

    jobs = []
    queued = {}

    for split, split_dir in split_sources.items():
        midi_files = sorted(list(split_dir.glob("*.mid")) + list(split_dir.glob("*.midi")))
        queued[split] = len(midi_files)
        print(f"queued {split}: {len(midi_files)} files", flush=True)

        for i, src in enumerate(midi_files):
            jobs.append((split, i, str(src.resolve())))

    counts = {"train": 0, "validation": 0, "test": 0}
    skipped = 0
    processed = 0
    total = len(jobs)

    write_status({
        "phase": "startup",
        "queued": queued,
        "processed": 0,
        "counts": counts,
        "skipped": skipped,
        "max_workers": MAX_WORKERS,
    })

    ctx = get_context("spawn")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
        futures = [ex.submit(tokenize_one, job) for job in jobs]

        for fut in as_completed(futures):
            result = fut.result()
            processed += 1

            if result["ok"]:
                counts[result["split"]] += 1
            else:
                skipped += 1
                print(f"skipped [{result['split']}] {result['file']}: {result['reason']}", flush=True)

            if processed % 250 == 0 or processed == total:
                print(
                    f"progress: {processed}/{total} | "
                    f"train={counts['train']} val={counts['validation']} test={counts['test']} "
                    f"skipped={skipped}",
                    flush=True,
                )
                write_status({
                    "phase": "tokenizing",
                    "queued": queued,
                    "processed": processed,
                    "total": total,
                    "counts": counts,
                    "skipped": skipped,
                    "max_workers": MAX_WORKERS,
                })

    meta = {
        "vocab_size": vocab_size,
        "counts": counts,
        "skipped": skipped,
        "tokenizer_path": str(TOK_PATH),
        "tokenization": "REMI",
        "augmentation": "train_transpose_-3_-2_-1_+1_+2_+3",
        "bpe": False,
        "parallel": True,
        "max_workers": MAX_WORKERS,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    write_status({
        "phase": "finished",
        "counts": counts,
        "skipped": skipped,
        "vocab_size": vocab_size,
        "max_workers": MAX_WORKERS,
    })

    print("done", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
