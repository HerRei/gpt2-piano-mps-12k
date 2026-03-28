from pathlib import Path
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
from miditok import REMI, TokenizerConfig

from _paths import ROOT

SPLITS = ROOT / "data" / "splits"

OUT = ROOT / "artifacts" / "tokens"
TOK_PATH = ROOT / "artifacts" / "tokenizer.json"
META_PATH = ROOT / "artifacts" / "meta.json"

# While your old training is still running, keep this modest.
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
    # Save tokenizer metadata once from the main process
    tokenizer = build_tokenizer()
    tokenizer.save(str(TOK_PATH))
    vocab_size = int(getattr(tokenizer, "vocab_size", len(tokenizer)))

    jobs = []
    for split in ["train", "validation", "test"]:
        split_dir = SPLITS / split
        midi_files = sorted(list(split_dir.glob("*.mid")) + list(split_dir.glob("*.midi")))
        print(f"queued {split}: {len(midi_files)} files", flush=True)

        for i, src in enumerate(midi_files):
            jobs.append((split, i, str(src.resolve())))

    counts = {"train": 0, "validation": 0, "test": 0}
    skipped = 0
    processed = 0
    total = len(jobs)

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
                print(
                    f"skipped [{result['split']}] {result['file']}: {result['reason']}",
                    flush=True,
                )

            if processed % 250 == 0 or processed == total:
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
        "tokenizer_path": str(TOK_PATH),
        "tokenization": "REMI",
        "augmentation": False,
        "bpe": False,
        "parallel": True,
        "max_workers": MAX_WORKERS,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print("done", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
