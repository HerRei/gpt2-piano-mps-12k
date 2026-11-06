"""Split MAESTRO, transpose the training pieces and tokenize everything with REMI.

usage: python v1/prepare_data.py --maestro path/to/maestro-v3.0.0 --out data/v1

The split is random by file (seed 42, 90/5/5), not MAESTRO's official split, so related performances can
land on both sides. Only training files are transposed, by -3..+3 semitones, and only when every note stays
on the piano. Writes <out>/{train,validation,test}/*.npy (int32 token ids), tokenizer.json and meta.json.
"""
import argparse
import json
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pretty_midi

TRANSPOSE = (-3, -2, -1, 1, 2, 3)
PIANO_LOW, PIANO_HIGH = 21, 108
MIN_TOKENS = 32


def remi():
    from miditok import REMI, TokenizerConfig

    return REMI(TokenizerConfig(
        use_programs=False,
        one_token_stream_for_programs=True,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_sustain_pedals=True,
        num_velocities=32,
        beat_res={(0, 4): 8, (4, 12): 4},
    ))


def ids_of(tokenizer, path):
    tok = tokenizer(str(path))
    if isinstance(tok, list):
        if len(tok) != 1:
            raise ValueError(f"expected one token stream, got {len(tok)}")
        tok = tok[0]
    return np.asarray(tok.ids, dtype=np.int32)


def transposed(src, semitones, tmp_dir):
    pm = pretty_midi.PrettyMIDI(str(src))
    pitches = [n.pitch for inst in pm.instruments for n in inst.notes]
    if not pitches or min(pitches) + semitones < PIANO_LOW or max(pitches) + semitones > PIANO_HIGH:
        return None
    for inst in pm.instruments:
        for n in inst.notes:
            n.pitch += semitones
    dst = Path(tmp_dir) / f"{src.stem}_{semitones:+d}.mid"
    pm.write(str(dst))
    return dst


def job(args):
    split, index, src, out_dir = args
    src, out_dir = Path(src), Path(out_dir)
    tokenizer = remi()
    variants = [("orig", src)]
    written = 0
    with tempfile.TemporaryDirectory() as tmp:
        if split == "train":
            for st in TRANSPOSE:
                path = transposed(src, st, tmp)
                if path is not None:
                    variants.append((f"tr{st:+d}", path))
        for tag, path in variants:
            ids = ids_of(tokenizer, path)
            if len(ids) < MIN_TOKENS:
                continue
            np.save(out_dir / split / f"{index:05d}_{tag}_{src.stem}.npy", ids)
            written += 1
    return split, written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maestro", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    files = sorted(list(a.maestro.rglob("*.mid")) + list(a.maestro.rglob("*.midi")))
    if not files:
        raise SystemExit(f"no MIDI files under {a.maestro}")
    random.seed(a.seed)
    random.shuffle(files)
    n_train, n_val = int(len(files) * 0.90), int(len(files) * 0.05)
    splits = {"train": files[:n_train], "validation": files[n_train:n_train + n_val],
              "test": files[n_train + n_val:]}
    for s in splits:
        (a.out / s).mkdir(parents=True, exist_ok=True)

    jobs = [(s, i, str(f), str(a.out)) for s, fs in splits.items() for i, f in enumerate(fs)]
    counts = {s: 0 for s in splits}
    with ProcessPoolExecutor(a.workers) as ex:
        for done, (s, n) in enumerate(ex.map(job, jobs, chunksize=4), 1):
            counts[s] += n
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} files", flush=True)

    tokenizer = remi()
    tokenizer.save(str(a.out / "tokenizer.json"))
    meta = {"vocab_size": len(tokenizer), "seed": a.seed, "files": {s: len(fs) for s, fs in splits.items()},
            "sequences": counts}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
