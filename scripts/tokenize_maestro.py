from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from _paths import ROOT
from tokenizer_utils import build_remi_tokenizer, token_ids_array

SPLITS = ROOT / "data" / "splits"

OUT = ROOT / "artifacts" / "tokens"
TOK_PATH = ROOT / "artifacts" / "tokenizer.json"
META_PATH = ROOT / "artifacts" / "meta.json"

# While your old training is still running, keep this modest.
MAX_WORKERS = 8

for split in ["train", "validation", "test"]:
    (OUT / split).mkdir(parents=True, exist_ok=True)


def tokenize_one(job):
    split, idx, src_str = job
    src = Path(src_str)
    out_dir = OUT / split

    try:
        tokenizer = build_remi_tokenizer()
        ids = token_ids_array(tokenizer(str(src)))

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
    tokenizer = build_remi_tokenizer()
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
        "single_token_stream_required": True,
        "parallel": True,
        "max_workers": MAX_WORKERS,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print("done", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
